import pandas as pd
import Assignment1.preprocessing_functions as pp

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

date_columns = ['property_scraped_at', 'host_since', 'reviews_first', 'reviews_last']
train_df, test_df = pp.fetch_data(date_columns)

# Transform 'property_last_updated' to a datetime object
pp.text_to_date_columns(train_df, 'property_scraped_at', ['property_last_updated'])
