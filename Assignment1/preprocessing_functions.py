from datetime import timedelta
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.neighbors
from sklearn.model_selection import train_test_split


# Fetch the data from the csv files parse the inputted date columns
def fetch_data(columns):
    train_df = pd.read_csv('train.csv', parse_dates=columns)

    y = train_df['target']
    X = train_df.drop(['target'], axis=1)

    return X, y


# Split the data into a train and test set
def split_train_test(X, y, test_size):
    train_df, test_df, train_target, test_target = sklearn.model_selection.train_test_split(X, y, test_size=test_size)
    return train_df, test_df, train_target, test_target


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
            else:
                datetime_date = timedelta()
            my_list.append([mytext, datetime_date])
        df[text_column] = df[text_column].apply(lambda x: dict(my_list)[x])
        df[text_column + '_dt'] = df[date_column] + df[text_column]
    return df


# Impute 'property_zipcode' missing values using a KNN imputer fitted on 'lat' and 'lon' columns
# Incorrect code, fix it?
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
