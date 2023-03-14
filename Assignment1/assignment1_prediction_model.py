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


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

data = pd.DataFrame(pd.read_csv('Assignment 1/data/assignment1_train.csv'))
target_variable = data['target']
data.drop(['target'], axis=1, inplace=True)
X = data
train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(X, target_variable, test_size=0.2)

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
train_data['property_zipcode_targetmean'] = train_data.apply(lambda row: zips_mean[str(row['property_zipcode'])], axis=1)
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

test_data['property_zipcode_targetmean'] = test_data.apply(lambda row: zips_mean[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else train_target.mean(), axis=1)
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


# PREDICTION
# train_data, test_data, train_target, test_target
def rmse(predictions, targets):
    rmse_val = pow(sum(pow(predictions - targets, 2)) / len(targets), 0.5)
    return(rmse_val)

# Keep only cleaned and usable columns, discard the rest
keep_columns = np.concatenate((amen_topN_list,extra_topN_list, ['location','property_bathrooms','property_bathrooms_missing','property_bedrooms','property_bedrooms_missing', 'property_zipcode_freq', 'property_zipcode_targetmean', 'property_feature_type0', 'property_feature_type1', 'property_feature_type2', 'property_feature_type3', 'property_lat', 'property_lon']))
train_data.drop(train_data.columns.difference(keep_columns), axis=1, inplace=True)
test_data.drop(test_data.columns.difference(keep_columns), axis=1, inplace=True)

# XGBOOST
xgb = xgboost.XGBRegressor(eta=0.05, max_depth=5, subsample=0.7)
xgb.fit(train_data, train_target)
xgb_preds_test = xgb.predict(test_data)
xgb_preds_train = xgb.predict(train_data)
print('mean test: ', rmse(train_target.mean(), test_target))
print('xgb test: ', rmse(xgb_preds_test, test_target))
print('mean train: ', rmse(train_target.mean(), train_target))
print('xgb train: ', rmse(xgb_preds_train, train_target))

# RANDOM FOREST
knn_properties = sklearn.ensemble.RandomForestRegressor(n_estimators=1000, max_depth=7, max_features=0.8, bootstrap=True)
knn_properties.fit(train_data.values, train_target.values)
preds = knn_properties.predict(test_data.values)
preds_train = knn_properties.predict(train_data.values)
print('mean test: ',rmse(train_target.mean(), test_target))
print('randomForest test: ',rmse(preds, test_target))
print('mean train: ',rmse(train_target.mean(), train_target))
print('randomForest train: ',rmse(preds_train, train_target))

# Sample output for train- and test data for purpose of comparison / validation that everything worked as inteded
print(train_data.head())
print(test_data.head())
