import pandas as pd
import Assignment1.preprocessing_functions as pp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

date_columns = ['property_scraped_at', 'host_since', 'reviews_first', 'reviews_last']
X, y = pp.fetch_data(date_columns)

# Perform train test split
train_df, test_df, train_target, test_target = pp.split_train_test(X, y, 0.2)

# Transform 'property_last_updated' to a datetime object
pp.text_to_date_columns(train_df, 'property_scraped_at', ['property_last_updated'])

# Calculate days passed since 'reviews_last' and 'reviews_first'
train_df = pp.days_passed(train_df, 'reviews_last', 'days_since_last_review')
train_df = pp.days_passed(train_df, 'reviews_first', 'days_since_first_review')

# INCORRECT
# Impute missing values in 'property_zipcode' using a KNN imputer fitted on 'lat' and 'lon' columns
#zipcode_imputer = pp.ZipcodeKNNImputer()
#zipcode_imputer.fit_transform(train_df)

# Create location variable indicating if location is in BRU or ANT (BRU = 1, ANT = 0)?
train_df = pp.BRU_or_ANT(train_df, 'property_zipcode')

