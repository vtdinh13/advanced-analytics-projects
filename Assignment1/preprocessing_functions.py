from datetime import timedelta
import pandas as pd
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def fetch_data(columns):
    train_df = pd.read_csv('train.csv', parse_dates=columns)
    test_df = pd.read_csv('test.csv', parse_dates=columns)
    return train_df, test_df


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
