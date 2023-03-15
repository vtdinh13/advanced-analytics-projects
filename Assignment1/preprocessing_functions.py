from datetime import timedelta
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.neighbors
from sklearn.model_selection import train_test_split


# Fetch the data from the csv files parse the inputted date columns
def fetch_data(date_col, cat_col):
    train_df = pd.read_csv('train.csv', parse_dates=date_col)

    # For all categorical columns, delete any spaces in the column values and make string values lowercase
    for col in cat_col:
        # WARNING: also deletes spaces between words of sentences
        train_df[col] = train_df[col].str.replace(' ', '')
        train_df[col] = train_df[col].str.lower()

    train_df[cat_col] = train_df[cat_col].astype('category')

    return train_df


# Split the data into a train and test set
def split_train_test(X, test_size):
    train_df, test_df = sklearn.model_selection.train_test_split(X, test_size=test_size, random_state=42)

    return train_df, test_df


# Function to transform the text in the 'property_last_updated' column to an integer
def string_to_int(input_var):
    return_number = 0
    if input_var[0] == 'a':
        return_number = 1
    else:
        return_number = int(input_var.split(" ", 1)[0])
    return return_number


# Transform the last_updated_text to a datetime object
def text_to_date_columns(df, date_column, text_columns):
    for text_column in text_columns:
        last_updated_text = df[text_column].unique()
        my_list = []
        for mytext in last_updated_text:
            if "week" in mytext:
                xyz = string_to_int(mytext)
                datetime_date = timedelta(weeks=-xyz)
            elif "month" in mytext:
                xyz = string_to_int(mytext) * 4 + 2
                datetime_date = timedelta(weeks=-xyz)
            elif "days" in mytext:
                xyz = string_to_int(mytext)
                datetime_date = timedelta(days=-xyz)
            elif mytext == "yesterday":
                datetime_date = timedelta(days=-1)
            elif mytext == "never":
                datetime_date = None
            else:
                datetime_date = timedelta()
            my_list.append([mytext, datetime_date])
        df[text_column] = df[text_column].apply(lambda x: dict(my_list)[x])
        df[text_column + '_dt'] = df[date_column] + df[text_column]
    return df


# Calculate days passed since the inputted date (column in date format) and 'property_scraped_at' date
def days_passed(df, date_column, new_column_name):
    df[new_column_name] = df['property_scraped_at'] - df[date_column]
    return df


# INCORRECT
# Impute 'property_zipcode' missing values using a KNN imputer fitted on 'lat' and 'lon' columns
class ZipcodeKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=4):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self.imputer = sklearn.neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors)
        self.imputer.fit(X[['property_lat', 'property_lon']], X['property_zipcode'])
        return self

    def transform(self, X, y=None):
        X['property_zipcode'] = self.imputer.predict(X[['property_lat', 'property_lon']])
        return X


# Create location variable indicating if location is in BRU or ANT using the 'property_zipcode' column, if 'property_zipcode' starts with 1, then the location is in BRU, otherwise it is in ANT
def BRU_or_ANT(df, zipcode):
    df['location'] = df[zipcode].apply(lambda x: 1 if str(x)[0] == '1' else 0)
    return df


# Function for encoding "property_feature_type"
def prop_type_bins(t):
    if t in ['Apartment']:
        return 'apartment'
    if t in ['House']:
        return 'house'
    if t in ['Loft', 'Bed & Breakfast']:
        return 'loft_and_bdnbrfst'
    if t in ['Townhouse', 'Guesthouse', 'Condominium', 'Other']:
        return 'town_guest_condo_other'
    else:
        return 'other'


# Function creating a variable counting the frequency of categorical variables
def count_freq(df, col):
    df[col + '_count'] = df.groupby(col)[col].transform('count')
    return df


# Function creating a variable attributing the mean target value to each categorical variable
def mean_target(df, col, target):
    df[col + '_mean_target'] = df.groupby(col)[target].transform('mean')
    return df
