import pandas as pd
import Assignment1.preprocessing_functions as pp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

date_columns = ['property_scraped_at', 'host_since', 'reviews_first', 'reviews_last']
categorical_columns = ['property_type', 'property_zipcode', 'property_room_type', 'property_bed_type', 'host_response_time', 'booking_cancel_policy']
X = pp.fetch_data(date_columns, categorical_columns)

# Perform train test split
train_df, test_df = pp.split_train_test(X, 0.2)

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

# Create 'property_feature_type' recoding 'property_type' into 3 categories: 'house', 'apartment', 'other'
train_df = train_df.assign(property_feature_type=train_df.property_type.map(pp.property_type_bins))

# Create variable giving the frequency of each zipcode
train_df = pp.count_freq(train_df, 'property_zipcode')

# Create variable attributing mean target to each zipcode
train_df = pp.mean_target(train_df, 'property_zipcode', 'target')

# One hot encode 'property_feature_type'
#train_df = pp.one_hot_encode(train_df, 'property_feature_type')

