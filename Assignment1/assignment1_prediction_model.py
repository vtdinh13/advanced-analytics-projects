import csv
import random
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import xgboost
import sklearn
from sklearn import ensemble
from sklearn import impute
from sklearn.neighbors import KNeighborsClassifier
import featuretools as ft
import missingno as msno
import ydata_profiling as yp
import datetime
from sklearn.preprocessing import MultiLabelBinarizer
import category_encoders as ce
from sklearn.feature_extraction import FeatureHasher
import statistics

pd.set_option('display.max_rows', 7000)
pd.set_option('display.max_columns', 7000)
pd.set_option('display.width', 7000)


data = pd.DataFrame(pd.read_csv('Assignment 1/data/assignment1_train.csv'))
TEST_DATA_SET = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/Adv. Analytics in Bus./pythonProjects/Assignment 1/data/test.csv'))
TEST_DATA_SET_property_id = TEST_DATA_SET['property_id']
data['property_lat'] = data['property_lat'].apply(lambda x: math.trunc(x*100)/100)
data['property_lon'] = data['property_lon'].apply(lambda x: math.trunc(x*100)/100)
#data['property_summary_len'] = data['property_summary'].apply(lambda x: len(str(x)))



#print(pd.concat([data['property_id'], data['target']], axis=1))
target_variable = data['target']
#data.drop(['target'], axis=1, inplace=True)
X = data
train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(X, target_variable, test_size=0.2)

#print(train_data['property_id'].iloc[range(20)])
#print(train_target.iloc[range(20)])

#pd.DataFrame(pd.concat([train_data['property_name'],train_target], axis=1)).to_csv('property_name_target.csv')
# Preprocessing Pipeline for TRAIN data
# Impute "property_zipcode"
train_data['property_zipcode'] = train_data['property_zipcode'].astype('category')
train_data['property_zipcode'] = train_data['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
zipcode_dataset = pd.DataFrame({'lat': train_data['property_lat'], 'lon': train_data['property_lon'], 'zipcode': train_data['property_zipcode']})
zipcode_dataset_noNA = zipcode_dataset[~zipcode_dataset.isna().any(axis=1)]
zipcode_dataset_noNA['lat'] = (zipcode_dataset_noNA['lat'] - zipcode_dataset_noNA['lat'].mean()) / np.std(zipcode_dataset_noNA['lat'])
zipcode_dataset_noNA['lon'] = (zipcode_dataset_noNA['lon'] - zipcode_dataset_noNA['lon'].mean()) / np.std(zipcode_dataset_noNA['lon'])
knn_imputer = KNeighborsClassifier(n_neighbors=4)
knn_imputer.fit(zipcode_dataset_noNA.iloc[:,:-1].values, zipcode_dataset_noNA.iloc[:,-1].values)
zipcode_dataset['lat'] = (zipcode_dataset['lat'] - zipcode_dataset['lat'].mean()) / np.std(zipcode_dataset['lat'])
zipcode_dataset['lon'] = (zipcode_dataset['lon'] - zipcode_dataset['lon'].mean()) / np.std(zipcode_dataset['lon'])
zipcode_dataset_imputed_zipcodes = zipcode_dataset.apply(lambda row: knn_imputer.predict([row.iloc[:-1]])[0] if pd.isna(row.iloc[-1]) else row.iloc[-1], axis=1)
train_data['property_zipcode'] = zipcode_dataset_imputed_zipcodes

# Create location variable indicating if location is in BRU or ANT
train_data['property_lat'] = (train_data['property_lat'] - train_data['property_lat'].mean()) / np.std(train_data['property_lat'])
train_data['property_lon'] = (train_data['property_lon'] - train_data['property_lon'].mean()) / np.std(train_data['property_lon'])
train_data['location'] = train_data['property_lat'].apply(lambda x: 1 if x < 1 else 0)

# Encode "property_zipcode"
zips_freq = train_data.groupby('property_zipcode')['property_zipcode'].count()
zips_with_target = pd.concat([train_data, train_target], axis=1)
zips_mean = zips_with_target.groupby('property_zipcode')['target'].mean()

# Instead of zipcode, add two columns. One with frequency, one with target mean
#train_data['property_zipcode_targetmean'] = train_data.apply(lambda row: zips_mean[str(row['property_zipcode'])], axis=1)
train_data['property_zipcode_freq'] = train_data.apply(lambda row: zips_freq[str(row['property_zipcode'])], axis=1)
zips_train = train_data['property_zipcode'].unique()

# Encode "property_type" and reduce number of bins
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

train_data['property_feature_type'] = train_data.apply(lambda x: prop_type_bins(x['property_type']),axis=1)

def hotOneEncode(dataset, columns):
    dummies = pd.DataFrame()
    encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
    for i in columns:
        encoder_df = pd.DataFrame(encoder.fit_transform(dataset[[str(i)]]).toarray())
        for j in range(len(encoder_df.columns)-1):
            dummies[str(i+str(j))] = encoder_df.iloc[:, j]
    return dummies

dums_train = hotOneEncode(train_data, ['property_feature_type'])
train_data.reset_index(drop=True, inplace=True)
dums_train.reset_index(drop=True, inplace=True)
train_data = pd.concat([train_data, dums_train], axis=1)

# Encode "property_amenities"
property_amenities = train_data['property_amenities']
for i, v in enumerate(property_amenities):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_amenities[i] = np.array(cleaner.split(','))

mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(property_amenities), columns=mlb.classes_)
amenities_freqs = pd.DataFrame(one_hot.sum())
amenities_freqs.columns = ['amenities_freqs']
amen_freq_sorted = amenities_freqs.sort_values('amenities_freqs', ascending=False)
# Indicate here how many of the top-n most frequently occurring amenities you want to keep
amen_topN_list = np.array(pd.DataFrame(amen_freq_sorted.iloc[range(10),:]).index)
train_data = pd.concat([train_data, one_hot], axis=1)

# Encode "extra"
property_extra = train_data['extra']
for i, v in enumerate(property_extra):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_extra[i] = np.array(cleaner.split(','))

one_hot = pd.DataFrame(mlb.fit_transform(property_extra), columns=mlb.classes_)
extra_freqs = pd.DataFrame(one_hot.sum())
extra_freqs.columns = ['extra_freqs']
extra_freq_sorted = extra_freqs.sort_values('extra_freqs', ascending=False)
# Indicate here how many of the top-n most frequently occurring "extra"-values you want to keep
extra_topN_list = np.array(pd.DataFrame(extra_freq_sorted.iloc[range(4),:]).index)
train_data = pd.concat([train_data, one_hot], axis=1)

# Impute values for features with nans or missing values
median_bathroom = train_data['property_bathrooms'].median()
train_data['property_bathrooms_missing'] = train_data['property_bathrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
train_data['property_bathrooms'] = train_data['property_bathrooms'].apply(lambda x: median_bathroom if pd.isna(x) else x)

median_bedroom = train_data['property_bedrooms'].median()
train_data['property_bedrooms_missing'] = train_data['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
train_data['property_bedrooms'] = train_data['property_bedrooms'].apply(lambda x: median_bedroom if pd.isna(x) else x)

# One-Hot encode "property_room_type"
property_roomtype = train_data['property_room_type']
for i, v in enumerate(property_roomtype):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_roomtype[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(property_roomtype), columns=mlb.classes_)
train_data = pd.concat([train_data, one_hot], axis=1)

# One-Hot encode "host_verified"
host_verified = train_data['host_verified']
for i, v in enumerate(host_verified):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    host_verified[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(host_verified), columns=mlb.classes_)
train_data['nr_verifications'] = one_hot.apply(lambda x: sum(x), axis=1)
#train_data = pd.concat([train_data, one_hot], axis=1)

# Include "reviews_X", since nan are correlated within reviews, all nan-values are replaced with 0 and replacement is indicated by a dummy "review_is_missing" which is 1 if it was missing, else 0
train_data['review_is_missing'] = train_data['reviews_value'].apply(lambda x: 1 if pd.isna(x) else 0)
reviews = train_data[['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month']]
train_data.drop(['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month'], axis=1, inplace=True)
reviews.fillna(0, inplace=True)
train_data = pd.concat([train_data, reviews], axis=1)

# Capture difference between "reviews_first" and "reviews_last" in new features "number of days between" and remove both features
def days_between(a, b):
    if not pd.isna(a):
        d1 = datetime.datetime.strptime(str(a), "%Y-%m-%d")
        d2 = datetime.datetime.strptime(str(b), "%Y-%m-%d")
        return abs((d2-d1).days)
    else:
        return -1

train_data['first_last_daydiff'] = train_data.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)

# Include "host_response_time" as ordinal feature
response_map = {"within an hour": 1, "within a few hours": 2, "within a day": 3, "a few days or more": 4}
train_data["response_time"] = train_data['host_response_time'].replace(response_map)
train_data["response_time"].fillna(5, inplace=True)

# Preprocessing Pipeline for TEST data
# Encode "property_zipcode" for TEST data
test_data['property_zipcode'] = test_data['property_zipcode'].astype('category')
test_data['property_zipcode'] = test_data['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
test_data['property_lat'] = (test_data['property_lat'] - test_data['property_lat'].mean()) / np.std(test_data['property_lat'])
test_data['property_lon'] = (test_data['property_lon'] - test_data['property_lon'].mean()) / np.std(test_data['property_lon'])
test_zipcodes_imputed = test_data.apply(lambda row: knn_imputer.predict([row.iloc[[12,13]]])[0] if pd.isna(row['property_zipcode']) else row['property_zipcode'], axis=1)
test_data['property_zipcode'] = test_zipcodes_imputed

# Encode location variable
test_data['location'] = test_data['property_lat'].apply(lambda x: 1 if x < 1 else 0)

#test_data['property_zipcode_targetmean'] = test_data.apply(lambda row: zips_mean[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else train_target.mean(), axis=1)
test_data['property_zipcode_freq'] = test_data.apply(lambda row: zips_freq[str(row['property_zipcode'])] if row['property_zipcode'] in zips_freq else 1, axis=1)

# Encode "property_feature" for TEST data
test_data['property_feature_type'] = test_data.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
dums_test = hotOneEncode(test_data, ['property_feature_type'])
test_data.reset_index(drop=True, inplace=True)
dums_test.reset_index(drop=True, inplace=True)
test_data = pd.concat([test_data, dums_test], axis=1)

# Encode "property_amenities" for TEST data
property_amenities = test_data['property_amenities']
for i, v in enumerate(property_amenities):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_amenities[i] = np.array(cleaner.split(','))

one_hot = pd.DataFrame(mlb.fit_transform(property_amenities), columns=mlb.classes_)
test_data = pd.concat([test_data, one_hot], axis=1)

# Encode "extra" for TEST data
property_extra = test_data['extra']
for i, v in enumerate(property_extra):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_extra[i] = np.array(cleaner.split(','))

one_hot = pd.DataFrame(mlb.fit_transform(property_extra), columns=mlb.classes_)
test_data = pd.concat([test_data, one_hot], axis=1)

# Impute values for features with nans or missing values for TEST data
test_data['property_bathrooms_missing'] = test_data['property_bathrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
test_data['property_bathrooms'] = test_data['property_bathrooms'].apply(lambda x: median_bathroom if pd.isna(x) else x)

test_data['property_bedrooms_missing'] = test_data['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
test_data['property_bedrooms'] = test_data['property_bedrooms'].apply(lambda x: median_bedroom if pd.isna(x) else x)

# One-Hot encode "property_room_type"
property_roomtype = test_data['property_room_type']
for i, v in enumerate(property_roomtype):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_roomtype[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(property_roomtype), columns=mlb.classes_)
test_data = pd.concat([test_data, one_hot], axis=1)

# One-Hot encode "host_verified"
host_verified = test_data['host_verified']
for i, v in enumerate(host_verified):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    host_verified[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(host_verified), columns=mlb.classes_)
test_data['nr_verifications'] = one_hot.apply(lambda x: sum(x), axis=1)
#test_data = pd.concat([test_data, one_hot], axis=1)

# Include "review_X"
test_data['review_is_missing'] = test_data['reviews_value'].apply(lambda x: 1 if pd.isna(x) else 0)
reviews = test_data[['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month']]
test_data.drop(['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month'], axis=1, inplace=True)
reviews.fillna(0, inplace=True)
test_data = pd.concat([test_data, reviews], axis=1)

# Include "first_last_daydiff" as the difference in days between "reviews_first" and "reviews_last"
test_data['first_last_daydiff'] = test_data.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)

# Include "host_response_time" as ordinal variable
test_data["response_time"] = test_data['host_response_time'].replace(response_map)
test_data["response_time"].fillna(5, inplace=True)








# PREPROCESSING FOR ACTUAL TEST SET
# Preprocessing Pipeline for TEST data
# Encode "property_zipcode" for TEST data
TEST_DATA_SET['property_zipcode'] = TEST_DATA_SET['property_zipcode'].astype('category')
TEST_DATA_SET['property_zipcode'] = TEST_DATA_SET['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
TEST_DATA_SET['property_lat'] = (TEST_DATA_SET['property_lat'] - TEST_DATA_SET['property_lat'].mean()) / np.std(TEST_DATA_SET['property_lat'])
TEST_DATA_SET['property_lon'] = (TEST_DATA_SET['property_lon'] - TEST_DATA_SET['property_lon'].mean()) / np.std(TEST_DATA_SET['property_lon'])
test_zipcodes_imputed = TEST_DATA_SET.apply(lambda row: knn_imputer.predict([row.iloc[[12,13]]])[0] if pd.isna(row['property_zipcode']) else row['property_zipcode'], axis=1)
TEST_DATA_SET['property_zipcode'] = test_zipcodes_imputed

# Encode location variable
TEST_DATA_SET['location'] = TEST_DATA_SET['property_lat'].apply(lambda x: 1 if x < 1 else 0)

#test_data['property_zipcode_targetmean'] = test_data.apply(lambda row: zips_mean[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else train_target.mean(), axis=1)
TEST_DATA_SET['property_zipcode_freq'] = TEST_DATA_SET.apply(lambda row: zips_freq[str(row['property_zipcode'])] if row['property_zipcode'] in zips_freq else 1, axis=1)

# Encode "property_feature" for TEST data
TEST_DATA_SET['property_feature_type'] = TEST_DATA_SET.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
dums_test = hotOneEncode(TEST_DATA_SET, ['property_feature_type'])
TEST_DATA_SET.reset_index(drop=True, inplace=True)
dums_test.reset_index(drop=True, inplace=True)
TEST_DATA_SET = pd.concat([TEST_DATA_SET, dums_test], axis=1)

# Encode "property_amenities" for TEST data
property_amenities = TEST_DATA_SET['property_amenities']
for i, v in enumerate(property_amenities):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_amenities[i] = np.array(cleaner.split(','))

one_hot = pd.DataFrame(mlb.fit_transform(property_amenities), columns=mlb.classes_)
TEST_DATA_SET = pd.concat([TEST_DATA_SET, one_hot], axis=1)

# Encode "extra" for TEST data
property_extra = TEST_DATA_SET['extra']
for i, v in enumerate(property_extra):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_extra[i] = np.array(cleaner.split(','))

one_hot = pd.DataFrame(mlb.fit_transform(property_extra), columns=mlb.classes_)
TEST_DATA_SET = pd.concat([TEST_DATA_SET, one_hot], axis=1)

# Impute values for features with nans or missing values for TEST data
TEST_DATA_SET['property_bathrooms_missing'] = TEST_DATA_SET['property_bathrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
TEST_DATA_SET['property_bathrooms'] = TEST_DATA_SET['property_bathrooms'].apply(lambda x: median_bathroom if pd.isna(x) else x)

TEST_DATA_SET['property_bedrooms_missing'] = TEST_DATA_SET['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
TEST_DATA_SET['property_bedrooms'] = TEST_DATA_SET['property_bedrooms'].apply(lambda x: median_bedroom if pd.isna(x) else x)

# One-Hot encode "property_room_type"
property_roomtype = TEST_DATA_SET['property_room_type']
for i, v in enumerate(property_roomtype):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    property_roomtype[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(property_roomtype), columns=mlb.classes_)
TEST_DATA_SET = pd.concat([TEST_DATA_SET, one_hot], axis=1)

# One-Hot encode "host_verified"
host_verified = TEST_DATA_SET['host_verified']
for i, v in enumerate(host_verified):
    cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
    host_verified[i] = np.array(cleaner.split(','))
one_hot = pd.DataFrame(mlb.fit_transform(host_verified), columns=mlb.classes_)
TEST_DATA_SET['nr_verifications'] = one_hot.apply(lambda x: sum(x), axis=1)
#test_data = pd.concat([test_data, one_hot], axis=1)

# Include "review_X"
TEST_DATA_SET['review_is_missing'] = TEST_DATA_SET['reviews_value'].apply(lambda x: 1 if pd.isna(x) else 0)
reviews = TEST_DATA_SET[['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month']]
TEST_DATA_SET.drop(['reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month'], axis=1, inplace=True)
reviews.fillna(0, inplace=True)
TEST_DATA_SET = pd.concat([TEST_DATA_SET, reviews], axis=1)

# Include "first_last_daydiff" as the difference in days between "reviews_first" and "reviews_last"
TEST_DATA_SET['first_last_daydiff'] = TEST_DATA_SET.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)

# Include "host_response_time" as ordinal variable
TEST_DATA_SET["response_time"] = TEST_DATA_SET['host_response_time'].replace(response_map)
TEST_DATA_SET["response_time"].fillna(5, inplace=True)









# PREDICTION
# train_data, test_data, train_target, test_target
def rmse(predictions, targets):
    rmse_val = pow(sum(pow(predictions - targets, 2)) / len(targets), 0.5)
    return(rmse_val)

def medianABsoluteError(p, t):
    medae = np.abs(np.array(t) - np.array(p))
    return statistics.median(medae)

def meanAbsoluteError(p, t):
    medae = np.abs(np.array(t) - np.array(p))
    return medae.mean()
# Keep only cleaned and usable columns, discard the rest

#test_if_ok = pd.concat([train_data['property_id'], train_target], axis=1)

#print(pd.concat([train_data['property_id'], train_target], axis=1).iloc[range(20),:])
#print("AFTER: ")
#print(train_data['property_id'].iloc[range(20)])
#print(train_target.iloc[range(20)])


# ['reviews_value', 'reviews_location', 'reviews_communication', 'reviews_cleanliness', 'reviews_acc', 'review_is_missing', 'property_bedrooms_missing', 'property_bathrooms_missing', 'IsLocationExact', 'HostHasProfilePic', 'Smokedetector', 'Kitchen', 'Heating', 'Familykidfriendly', 'reviews_num', 'property_bedrooms']
# np.concatenate((amen_topN_list,extra_topN_list,['nr_verifications', 'Privateroom', 'Sharedroom','location','property_bathrooms','property_bathrooms_missing','property_bedrooms','property_bedrooms_missing', 'property_zipcode_freq', 'property_feature_type0', 'property_feature_type1', 'property_feature_type2', 'property_feature_type3', 'property_lat', 'property_lon']
keep_columns = np.concatenate((amen_topN_list,extra_topN_list,['property_lat', 'property_lon', 'booking_price_covers', 'response_time', 'first_last_daydiff','reviews_num', 'review_is_missing', 'reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month', 'Privateroom', 'Sharedroom','location','property_bathrooms','property_bathrooms_missing','property_bedrooms','property_bedrooms_missing', 'property_zipcode_freq', 'target']))
keep_columns_2 = ['booking_price_covers', 'reviews_value', 'reviews_location', 'reviews_communication', 'reviews_cleanliness', 'reviews_acc', 'review_is_missing', 'property_bedrooms_missing', 'property_bathrooms_missing', 'IsLocationExact', 'HostHasProfilePic', 'Smokedetector', 'Kitchen', 'Heating', 'Familykidfriendly', 'reviews_num', 'property_bedrooms', 'target']
keep_columns_3 = [ 'Shampoo', 'target']
train_data.drop(train_data.columns.difference(keep_columns), axis=1, inplace=True)
test_data.drop(test_data.columns.difference(keep_columns), axis=1, inplace=True)

keep_columns_TEST = np.concatenate((amen_topN_list,extra_topN_list,['property_lat', 'property_lon', 'booking_price_covers', 'response_time', 'first_last_daydiff','reviews_num', 'review_is_missing', 'reviews_acc','reviews_cleanliness', 'reviews_checkin', 'reviews_communication', 'reviews_location', 'reviews_value','reviews_per_month', 'Privateroom', 'Sharedroom','location','property_bathrooms','property_bathrooms_missing','property_bedrooms','property_bedrooms_missing', 'property_zipcode_freq']))
TEST_DATA_SET.drop(TEST_DATA_SET.columns.difference(keep_columns_TEST), axis=1, inplace=True)



training_target = train_data['target']
train_data.drop(['target'], axis=1, inplace=True)

testing_target = test_data['target']
test_data.drop(['target'], axis=1, inplace=True)


xgb = xgboost.XGBRegressor(eta=0.02, max_depth=4, subsample=0.8)
xgb.fit(train_data, training_target)
xgb_preds_test = xgb.predict(test_data)
xgb_preds_train = xgb.predict(train_data)
xgb_predictions_for_test_set = xgb.predict(TEST_DATA_SET)

pd.concat([pd.DataFrame(TEST_DATA_SET_property_id), pd.DataFrame(xgb_predictions_for_test_set)], axis=1).to_csv("test_data_predictions.csv", index=False)
print('Mean-Pred- RMSE: ', rmse(training_target.mean(), testing_target)-rmse(xgb_preds_test, testing_target))
#print('TEST - RMSE: ', rmse(xgb_preds_test, testing_target))
#print('mean train: ', rmse(training_target.mean(), training_target))
#print('TRAIN - RMSE: ', rmse(xgb_preds_train, training_target))

print("Mean-Pred. - MedianAbsoluteError: ", medianABsoluteError(training_target.mean(), testing_target)-medianABsoluteError(xgb_preds_test, testing_target))
#print("TRAIN - MedianAbsoluteError: ", medianABsoluteError(xgb_preds_train, training_target))
#print("TEST - MedianAbsoluteError: ", medianABsoluteError(xgb_preds_test, testing_target))

print("Mean-Pred. - MeanAbsoluteError: ", meanAbsoluteError(training_target.mean(), testing_target)-meanAbsoluteError(xgb_preds_test, testing_target))
#print("TRAIN - MeanAbsoluteError: ", meanAbsoluteError(xgb_preds_train, training_target))
#print("TEST - MeanAbsoluteError: ", meanAbsoluteError(xgb_preds_test, testing_target))


#linear_model = sklearn.linear_model.LinearRegression()
#linear_model.fit(train_data.values, training_target.values)
#linear_preds_test = linear_model.predict(test_data.values)
#linear_preds_train = linear_model.predict(train_data.values)
#print('mean test: ', rmse(training_target.mean(),testing_target))
#print('linear test: ', rmse(linear_preds_test, testing_target))
#print('mean train: ', rmse(training_target.mean(), training_target))
#print('linear train: ', rmse(linear_preds_train, training_target))

def randf():
    # RANDOM FOREST
    knn_properties = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, max_depth=4, bootstrap=True, max_features=0.8)
    knn_properties.fit(train_data.values, training_target.values)
    preds = knn_properties.predict(test_data.values)
    preds_train = knn_properties.predict(train_data.values)
    print('mean test: ',rmse(train_target.mean(), testing_target))
    print('randomForest test: ',rmse(preds, testing_target))
    print('mean train: ',rmse(train_target.mean(), training_target))
    print('randomForest train: ',rmse(preds_train, training_target))
    print(" RFOREST Mean-Pred. Difference - MedianAbsoluteError: ", medianABsoluteError(training_target.mean(), testing_target)-medianABsoluteError(preds, testing_target))
    print("RFOREST Mean-Pred. Difference - MeanAbsoluteError: ", meanAbsoluteError(training_target.mean(), testing_target)-meanAbsoluteError(preds, testing_target))

    # Sample output for train- and test data for purpose of comparison / validation that everything worked as inteded
    #print(train_data.head())
    #print(train_target.head())
    #print(train_data[['reviews_value', 'reviews_location', 'reviews_communication', 'reviews_cleanliness', 'reviews_acc', 'review_is_missing', 'property_bedrooms_missing', 'property_bathrooms_missing', 'IsLocationExact', 'HostHasProfilePic', 'Smokedetector', 'Kitchen', 'Heating', 'Familykidfriendly', 'reviews_num', 'property_bedrooms']].corr())
