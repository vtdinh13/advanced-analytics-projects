import pandas as pd
import Assignment1.preprocessing_functions as pp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

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

# Recode 'property_type' into 5 categories: 'house', 'apartment', 'other'
# Mapping 'property_type' to 'property_feature_type'
property_type_bins = {'apartment': 'Apartment',
                      'house': 'House',
                      'villa': 'House',
                      'loft': 'Loft_and_B&B',
                      'bed&breakfast': 'Loft_and_B&B',
                      'townhouse': 'Town_Guest_Condo_Other',
                      'guesthouse': 'Town_Guest_Condo_Other',
                      'condominium': 'Town_Guest_Condo_Other',
                      'other': 'Town_Guest_Condo_Other',
                      'boat': 'Other',
                      'boutiquehotel': 'Other',
                      'cabin': 'Other',
                      'camper/rv': 'Other',
                      'castle': 'Other',
                      'chalet': 'Other',
                      'dorm': 'Other',
                      'earthhouse': 'Other',
                      'guestsuite': 'Other',
                      'hostel': 'Other',
                      'servicedapartment': 'Other',
                      'tent': 'Other',
                      'timeshare': 'Other',
                      'yurt': 'Other'}

train_df = pp.recode(train_df, 'property_type', property_type_bins)

# Create variable giving the frequency of each zipcode
train_df = pp.count_freq(train_df, 'property_zipcode')

# Create variable attributing mean target to each zipcode
train_df = pp.mean_target(train_df, 'property_zipcode', 'target')

# One hot encode 'property_feature_type'
train_df = pp.one_hot_encode(train_df, 'property_type_recoded')



