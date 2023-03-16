from datetime import timedelta
from geopy.geocoders import Nominatim
import pandas as pd
import os
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


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


# Function to impute missing values in 'property_zipcode' using latitude and longitude values
def get_zipcode(lat, lon):
    geolocator = Nominatim(user_agent="my_application")
    location = geolocator.reverse(f'{lat}, {lon}')
    address = location.raw['address']
    zipcode = address.get('postcode')
    return zipcode


def impute_zipcode(df, lat_col, lon_col, zipcode_col):
    df = df.copy()
    missing_zip_idx = df[df[zipcode_col].isnull()].index
    for idx in missing_zip_idx:
        lat = df.at[idx, lat_col]
        lon = df.at[idx, lon_col]
        zipcode = get_zipcode(lat, lon)
        df.at[idx, zipcode_col] = zipcode

    return df


# Custom transformer to impute missing values in 'property_zipcode' using latitude and longitude values
class ZipcodeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, lat, lon, zipcode_col):
        self.lat = lat
        self.lon = lon
        self.zipcode_col = zipcode_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        missing_zip_idx = X[X[self.zipcode_col].isnull()].index
        for idx in missing_zip_idx:
            lat = X.at[idx, self.lat]
            lon = X.at[idx, self.lon]
            zipcode = impute_zipcode(lat, lon)
            X.at[idx, self.zipcode_col] = zipcode

        return X


# Create location variable indicating if location is in BRU or ANT using the 'property_zipcode' column, if
# 'property_zipcode' starts with 1, then the location is in BRU, otherwise it is in ANT
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


# Recoding categorical variables
def recode(df, col, mapping):
    df = df.assign(**{f'{col}_recoded': df[col].map(mapping)})
    df[f'{col}_recoded'] = df[f'{col}_recoded'].astype('category')
    return df


# One hot encode categorical variables
def one_hot_encode(df, cat_col):
    categorical_encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_column = categorical_encoder.fit_transform(df[[cat_col]]).toarray()
    encoded_column = pd.DataFrame(encoded_column, columns=categorical_encoder.get_feature_names_out([cat_col]))
    df.reset_index(drop=True, inplace=True)
    encoded_column.reset_index(drop=True, inplace=True)
    df = pd.concat([df, encoded_column], axis=1)
    return df


# Convert columns with comma separated list to OneHotEncoded columns using the MultiLabelBinarizer and keep the top n
# most frequently occurring values as columns
def one_hot_encode_list_col(df, col, n='max'):
    mlb = MultiLabelBinarizer()
    df[col] = df[col].str.replace(' ', '')
    df[col] = df[col].str.lower()
    df[col] = df[col].fillna('unknown')
    one_hot_encoded_col = pd.DataFrame(mlb.fit_transform(df[col].str.split(',')), columns=mlb.classes_,
                                       index=df.property_id).reset_index().drop(columns='property_id')

    if n == 'max':
        return df
    else:
        col_freq = pd.DataFrame(one_hot_encoded_col.sum(), columns=[f'{col}_freq'])
        col_freq = col_freq.sort_values(by=f'{col}_freq', ascending=False)
        top_n_amen = np.array(pd.DataFrame(col_freq.iloc[range(n), :]).index)
        df = pd.concat([df, one_hot_encoded_col[top_n_amen]], axis=1)
        return df
