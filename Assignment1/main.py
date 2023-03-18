import pandas as pd
from feature_engine.imputation import CategoricalImputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from feature_engine.encoding import CountFrequencyEncoder, MeanEncoder
import Assignment1.preprocessing_functions as pp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

date_columns = ['property_scraped_at', 'host_since', 'reviews_first', 'reviews_last']
# 'property_zipcode' needs to be made categorical only AFTER imputation
categorical_columns = ['property_type', 'property_room_type', 'property_bed_type', 'host_response_time',
                       'booking_cancel_policy']
X = pp.fetch_data(date_columns, categorical_columns)

# Perform train test split
train_df, test_df = pp.split_train_test(X, 0.2)

## Create FunctionTransformers using preprocessing_functions.py functions
# Transform 'property_last_updated' to a datetime object
property_last_updated_transformer = FunctionTransformer(pp.text_to_date_columns, kw_args={'date_column': 'property_scraped_at', 'text_column': 'property_last_updated'})

# Calculate days passed since 'reviews_last' and 'reviews_first'
days_since_last_review_transformer = FunctionTransformer(pp.days_passed, kw_args={'date_column': 'reviews_last'})
days_since_first_review_transformer = FunctionTransformer(pp.days_passed, kw_args={'date_column': 'reviews_first'})

# Create a pipeline for the 'property_zipcode' column
property_zipcode_pipeline = Pipeline(steps=[
    ('zipcode_imputer', pp.ZipcodeImputer('property_lat', 'property_lon', 'property_zipcode')),
    ('leftover_zipcode_imputer', CategoricalImputer(imputation_method='frequent', variables=['property_zipcode'])),
    ('frequency_encoder', CountFrequencyEncoder(variables=['property_zipcode'])),
    #('mean_target_encoder', MeanEncoder(variables=['property_zipcode'])),
    ('BRU_or_ANT', FunctionTransformer(pp.BRU_or_ANT, kw_args={'zipcode': 'property_zipcode'}))])

preprocessing_pipeline = Pipeline(steps=[
    ('reset_index', FunctionTransformer(pp.reset_index)),
    ('property_last_updated_transformer', property_last_updated_transformer),
    ('days_since_last_review', FunctionTransformer(pp.days_passed, kw_args={'date_column': 'reviews_last'})),
    ('days_since_first_review', FunctionTransformer(pp.days_passed, kw_args={'date_column': 'reviews_first'})),
    ('property_zipcode_pipeline', property_zipcode_pipeline)])

train_df = preprocessing_pipeline.fit_transform(train_df, y=train_df['target'])
test_df = preprocessing_pipeline.transform(test_df)



# Impute missing values in 'property_zipcode' using latitude and longitude values
ZI = pp.ZipcodeImputer('property_lat', 'property_lon', 'property_zipcode')
train_df = ZI.fit_transform(train_df)

# Create variable giving the frequency of each zipcode
train_df = pp.count_freq(train_df, 'property_zipcode')

# Create variable attributing mean target to each zipcode
train_df = pp.mean_target(train_df, 'property_zipcode', 'target')

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

# One hot encode 'property_feature_type'
train_df = pp.one_hot_encode(train_df, 'property_type_recoded')

# Convert 'property_amenities' and 'extras' to one hot encoded columns of the top n most frequently occurring values
train_df = pp.one_hot_encode_list_col(train_df, 'property_amenities', n=10)
train_df = pp.one_hot_encode_list_col(train_df, 'extra', n=4)

# Impute missing values in 'property_bedrooms', 'property_beds' and 'property_bathrooms' using the median
SI = SimpleImputer(strategy='median')
train_df['property_bedrooms'] = SI.fit_transform(train_df[['property_bedrooms']])
train_df['property_bathrooms'] = SI.fit_transform(train_df[['property_bathrooms']])
train_df['property_beds'] = SI.fit_transform(train_df[['property_beds']])
