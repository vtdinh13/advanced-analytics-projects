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
import warnings
from xgboost import plot_importance
from autoimpute.imputations import SingleImputer, MultipleImputer
from sklearn.ensemble import IsolationForest


# Final prediction model for AirBnB Data for Assignment 1
pd.set_option('display.max_rows', 7000)
pd.set_option('display.max_columns', 7000)
pd.set_option('display.width', 7000)

# Import of pre-separated train-test split data
X_train = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/TrainTestSplitData/X_train.csv'))
X_valid = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/TrainTestSplitData/X_test.csv'))
y_train = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/TrainTestSplitData/y_train.csv'))
y_valid = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/TrainTestSplitData/y_test.csv'))
X_test = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/test_data.csv'))
clustering = pd.DataFrame(pd.read_excel('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/cluster_results_df.xlsx'))
clustering.drop(['property_lat'], axis=1, inplace=True)
clustering.drop(['property_lon'], axis=1, inplace=True)
X_train = pd.merge(X_train, clustering, on='property_id', how='left')
X_train.drop(['city'], inplace=True, axis=1)


# Log-transform target => NOT done as it doesn't improve performance
#X_train['target'] = np.log(X_train['target'])
#X_valid['target'] = np.log(X_valid['target'])
#y_train = np.log(y_train)
#y_valid = np.log(y_valid)

def rmse(p, t, exp):
    if exp==True:
        p = np.exp(p)
        t = np.exp(t)
    p = np.array(p)
    t = np.array(t)
    rmse_val = pow(sum(pow(p - t, 2)) / len(t), 0.5)
    return(rmse_val[0])

def meanAbsoluteError(p, t, exp):
    if exp==True:
        p = np.exp(p)
        t = np.exp(t)
    p = np.array(p)
    t = np.array(t)
    medae = np.abs(np.array(t) - np.array(p))
    return medae.mean()

def convert_coordinates(lat, lon):
    LL = pd.concat([lat, lon], axis=1)
    LL.columns = ['lat', 'lon']
    lat_converted = LL.apply(lambda x: math.sqrt(math.pow(x['lat'], 2) + math.pow(x['lon'], 2)), axis=1)
    lon_converted = LL.apply(lambda x: np.arctan(x['lon']/x['lat']), axis=1)
    return lat_converted, lon_converted

def standardize(values, basis):
    return ((values-basis.mean())/np.std(basis))

def hotOneEncode(dataset, columns):
    dummies = pd.DataFrame()
    encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
    for i in columns:
        encoder_df = pd.DataFrame(encoder.fit_transform(dataset[[str(i)]]).toarray())
        for j in range(len(encoder_df.columns)-1):
            dummies[str(i+str(j))] = encoder_df.iloc[:, j]
    return dummies

def infer_cluster(lat, lon, cluster_means):
    cords = pd.concat([lat, lon], axis=1)
    cords.columns = ['lat', 'lon']
    cluster_ids = []
    for idx, i in cords.iterrows():
        dist = []
        for didx, d in cluster_means.iterrows():
            dist.append(np.sqrt(
                pow(d['lat']-i['lat'], 2)+pow(d['lon']-i['lon'], 2)
            ))
        cluster_ids.append(dist.index(min(dist)))
    return cluster_ids

def infer_cluster_singles(lat, lon, cluster_means):
    dist = []
    for didx, d in cluster_means.iterrows():
        dist.append(np.sqrt(
            pow(d['lat']-lat, 2)+pow(d['lon']-lon, 2)
        ))
    cluster_id = dist.index(min(dist))
    return cluster_id

def last_updated_in_days(luid_col):
    luid = []
    map_time = {'day': 1, 'days': 1, 'week': 7, 'weeks': 7, 'month': 30, 'months': 30, 'never': -1, 'yesterday': 1, 'today': 0, 'a': 1}
    for idx, i in enumerate(luid_col):
        i_array = pd.DataFrame(i.split())
        if len(i_array) > 1:
            if i_array.iloc[0][0] == 'a':
                val = i_array.iloc[1].replace(map_time)
                luid.append(val[0])
            else:
                val = int(i_array.iloc[0][0]) * i_array.iloc[1].replace(map_time)
                luid.append(val[0])
        else:
            val = i_array.iloc[0].replace(map_time)
            luid.append(val[0])
    return np.array(luid)

def binarize_amenities(data_amenities, keep_top_n, master):
    mlb = MultiLabelBinarizer()
    property_amenities = data_amenities
    for i, v in enumerate(property_amenities):
        cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
        property_amenities[i] = np.array(cleaner.split(','))
    one_hot_amenities = pd.DataFrame(mlb.fit_transform(property_amenities), columns=mlb.classes_)
    if master==True:
        amenities_freqs = pd.DataFrame(one_hot_amenities.sum())
        amenities_freqs.columns = ['amenities_freqs']
        amen_freq_sorted = amenities_freqs.sort_values('amenities_freqs', ascending=False)
        amen_topN_list = np.array(pd.DataFrame(amen_freq_sorted.iloc[range(keep_top_n), :]).index)
        #print(amen_topN_list)
        one_hot_amenities.drop(one_hot_amenities.columns.difference(amen_topN_list), axis=1, inplace=True)
        return one_hot_amenities, amen_topN_list
    else:
        return one_hot_amenities

def verified_nr(col):
    mlb = MultiLabelBinarizer()
    host_verified = col
    for i, v in enumerate(host_verified):
        cleaner = str(v).replace('"', "").replace("'", "").replace(" ", "").replace("/", "").replace("]", "").replace("[", "")
        host_verified[i] = np.array(cleaner.split(','))
    one_hot = pd.DataFrame(mlb.fit_transform(host_verified), columns=mlb.classes_)
    one_hot_sum = one_hot.apply(lambda x: sum(x), axis=1)
    return one_hot_sum

# Transform polar coordinates to cartesian coordinates
X_train['property_lat'], X_train['property_lon'] = convert_coordinates(X_train['property_lat'], X_train['property_lon'])
X_valid['property_lat'], X_valid['property_lon'] = convert_coordinates(X_valid['property_lat'], X_valid['property_lon'])
X_test['property_lat'], X_test['property_lon'] = convert_coordinates(X_test['property_lat'], X_test['property_lon'])

# Infer cluster_id
cluster_lat = X_train.groupby('cluster_id')['property_lat'].mean()
cluster_lon = X_train.groupby('cluster_id')['property_lon'].mean()
cluster_cords = pd.concat([cluster_lat, cluster_lon], axis=1)
cluster_cords.columns = ['lat', 'lon']
#X_train['cluster_id'] = infer_cluster(X_train['property_lat'], X_train['property_lon'], cluster_cords)
#X_train['cluster_id'] = X_train.apply(lambda x: infer_cluster_singles(x['property_lat'], x['property_lon'], cluster_cords) if pd.isna(x['cluster_id']) else x['cluster_id'], axis=1)
X_valid['cluster_id'] = infer_cluster(X_valid['property_lat'], X_valid['property_lon'], cluster_cords)
X_test['cluster_id'] = infer_cluster(X_test['property_lat'], X_test['property_lon'], cluster_cords)

# Mean-target encode cluster_id
#cluster_target_mean = X_train[['cluster_id', 'target']].groupby('cluster_id')['target'].mean()
#impute_clusterId = SingleImputer(strategy={'cluster_id':'median'}, predictors=['property_lat', 'property_lon'])
#impute_clusterId.fit(X_train)
#X_train = impute_clusterId.transform(X_train)
#X_train['cluster_id'] = X_train.apply(lambda row: cluster_target_mean.iloc[int(row['cluster_id'])], axis=1)
#X_valid['cluster_id'] = X_valid.apply(lambda row: cluster_target_mean.iloc[int(row['cluster_id'])], axis=1)
#X_test['cluster_id'] = X_test.apply(lambda row: cluster_target_mean.iloc[int(row['cluster_id'])], axis=1)



# Impute missing zipcode values from X_train
zipcode_knn_imputer = KNeighborsClassifier(n_neighbors=4)
X_train['property_zipcode'] = X_train['property_zipcode'].astype('category')
X_train['property_zipcode'] = X_train['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
X_train_zipcodes_with_na = pd.DataFrame({'lat': X_train['property_lat'], 'lon': X_train['property_lon'], 'zipcode': X_train['property_zipcode']})
X_train_zipcodes_without_na = X_train_zipcodes_with_na[~X_train_zipcodes_with_na.isna().any(axis=1)]
X_train_zipcodes_without_na['lat'] = standardize(X_train_zipcodes_without_na['lat'], X_train_zipcodes_without_na['lat'])
X_train_zipcodes_without_na['lon'] = standardize(X_train_zipcodes_without_na['lon'], X_train_zipcodes_without_na['lon'])
zipcode_knn_imputer.fit(X_train_zipcodes_without_na.iloc[:,:-1].values, X_train_zipcodes_without_na.iloc[:,-1].values)
X_train_zipcodes_with_na['lat'] = standardize(X_train_zipcodes_with_na['lat'], X_train_zipcodes_with_na['lat'])
X_train_zipcodes_with_na['lon'] = standardize(X_train_zipcodes_with_na['lon'], X_train_zipcodes_with_na['lon'])
X_train_imputed_zipcodes = X_train_zipcodes_with_na.apply(lambda row: zipcode_knn_imputer.predict([row.iloc[:-1]])[0] if pd.isna(row.iloc[-1]) else row.iloc[-1], axis=1)
X_train['property_zipcode'] = X_train_imputed_zipcodes
# Apply zipcode_knn_imputer on X_valid
X_valid['property_zipcode'] = X_valid['property_zipcode'].astype('category')
X_valid['property_zipcode'] = X_valid['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
X_valid['property_lat'] = standardize(X_valid['property_lat'], X_train['property_lat'])
X_valid['property_lon'] = standardize(X_valid['property_lon'], X_train['property_lon'])
X_valid_imputed_zipcodes = X_valid.apply(lambda row: zipcode_knn_imputer.predict([row.iloc[[12,13]]])[0] if pd.isna(row['property_zipcode']) else row['property_zipcode'], axis=1)
X_valid['property_zipcode'] = X_valid_imputed_zipcodes
# Apply zipcode_knn_imputer on X_valid
X_test['property_zipcode'] = X_test['property_zipcode'].astype('category')
X_test['property_zipcode'] = X_test['property_zipcode'].apply(lambda x: str(x).replace(" ", ""))
X_test['property_lat'] = standardize(X_test['property_lat'], X_train['property_lat'])
X_test['property_lon'] = standardize(X_test['property_lon'], X_train['property_lon'])
X_test_imputed_zipcodes = X_test.apply(lambda row: zipcode_knn_imputer.predict([row.iloc[[12,13]]])[0] if pd.isna(row['property_zipcode']) else row['property_zipcode'], axis=1)
X_test['property_zipcode'] = X_valid_imputed_zipcodes

# Add a column indicating if listing is in BRU or ANT
X_train['city'] = X_train['property_lat'].apply(lambda x: 1 if x < 51.2 else 0)
X_valid['city'] = X_valid['property_lat'].apply(lambda x: 1 if x < 51.2 else 0)
X_test['city'] = X_test['property_lat'].apply(lambda x: 1 if x < 51.2 else 0)


# Add a column with zipcode frequency
zips_freq = X_train.groupby('property_zipcode')['property_zipcode'].count()
zips_with_target = pd.concat([X_train, y_train], axis=1)
zips_mean = zips_with_target.groupby('property_zipcode')['target'].mean()
X_train['property_zipcode_freq'] = X_train.apply(lambda row: zips_freq[str(row['property_zipcode'])], axis=1)
X_valid['property_zipcode_freq'] = X_valid.apply(lambda row: zips_freq[str(row['property_zipcode'])] if row['property_zipcode'] in zips_freq else 1, axis=1)
X_test['property_zipcode_freq'] = X_test.apply(lambda row: zips_freq[str(row['property_zipcode'])] if row['property_zipcode'] in zips_freq else 1, axis=1)

# Zipcode Target Mean Column
zips_train = X_train['property_zipcode'].unique()
zips_mean = pd.DataFrame(zips_with_target.groupby('property_zipcode')['target'].mean()).iloc[:,-1]

#X_train['property_zipcode_targetmean'] = X_train.apply(lambda row: zips_mean[str(row['property_zipcode'])], axis=1)
X_train['property_zipcode_targetmean'] = X_train.apply(lambda row: zips_mean.loc[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else y_train.mean(), axis=1)
X_valid['property_zipcode_targetmean'] = X_valid.apply(lambda row: zips_mean.loc[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else y_train.mean(), axis=1)
X_test['property_zipcode_targetmean'] = X_test.apply(lambda row: zips_mean.loc[str(row['property_zipcode'])] if row['property_zipcode'] in zips_train else y_train.mean(), axis=1)

# Encode property_type and bin it
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

X_train['property_feature_type'] = X_train.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
X_train = pd.concat([X_train, hotOneEncode(X_train, ['property_feature_type'])], axis=1)

X_valid['property_feature_type'] = X_valid.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
X_valid = pd.concat([X_valid, hotOneEncode(X_valid, ['property_feature_type'])], axis=1)

X_test['property_feature_type'] = X_test.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
X_test = pd.concat([X_test, hotOneEncode(X_test, ['property_feature_type'])], axis=1)

# Impute Reviews
#'reviews_rating','reviews_cleanliness',  'reviews_location', 'reviews_value'
impute_reviews_acc = SingleImputer(strategy={'reviews_acc':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_rating',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_location',
                                                                                         'reviews_value'])

impute_reviews_acc.fit(X_train)
X_train = impute_reviews_acc.transform(X_train)
X_valid = impute_reviews_acc.transform(X_valid)
X_test = impute_reviews_acc.transform(X_test)

impute_reviews_rating = SingleImputer(strategy={'reviews_rating':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_acc',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_location',
                                                                                         'reviews_value'])

impute_reviews_rating.fit(X_train)
X_train = impute_reviews_rating.transform(X_train)
X_valid = impute_reviews_rating.transform(X_valid)
X_test = impute_reviews_rating.transform(X_test)

impute_reviews_location = SingleImputer(strategy={'reviews_location':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_acc',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_rating',
                                                                                         'reviews_value'])

impute_reviews_location.fit(X_train)
X_train = impute_reviews_location.transform(X_train)
X_valid = impute_reviews_location.transform(X_valid)
X_test = impute_reviews_location.transform(X_test)

impute_reviews_value = SingleImputer(strategy={'reviews_value':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_acc',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_rating',
                                                                                         'reviews_location'])

impute_reviews_value.fit(X_train)
X_train = impute_reviews_value.transform(X_train)
X_valid = impute_reviews_value.transform(X_valid)
X_test = impute_reviews_value.transform(X_test)

impute_reviews_cleanliness = SingleImputer(strategy={'reviews_cleanliness':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_acc',
                                                                                         'reviews_value',
                                                                                         'reviews_checkin',
                                                                                         'reviews_rating',
                                                                                         'reviews_location'])

impute_reviews_cleanliness.fit(X_train)
X_train = impute_reviews_cleanliness.transform(X_train)
X_valid = impute_reviews_cleanliness.transform(X_valid)
X_test = impute_reviews_cleanliness.transform(X_test)

# Encode "last_updated_on"
X_train['property_last_updated'] = last_updated_in_days(X_train['property_last_updated'])
X_valid['property_last_updated'] = last_updated_in_days(X_valid['property_last_updated'])
X_test['property_last_updated'] = last_updated_in_days(X_test['property_last_updated'])

# Standardize "property_last_updated"
#X_train['property_last_updated'] = standardize(X_train['property_last_updated'], X_train['property_last_updated'])
#X_valid['property_last_updated'] = standardize(X_valid['property_last_updated'], X_train['property_last_updated'])
#X_test['property_last_updated'] = standardize(X_test['property_last_updated'], X_train['property_last_updated'])

# without standardization: 45.83292936772679
# with standardization: 46.00769874018439

# Encode bedrooms
train_median_bedroom = X_train['property_bedrooms'].mean()
X_train['property_bedrooms_missing'] = X_train['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
X_train['property_bedrooms'] = X_train['property_bedrooms'].apply(lambda x: train_median_bedroom if pd.isna(x) else x)
X_valid['property_bedrooms_missing'] = X_valid['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
X_valid['property_bedrooms'] = X_valid['property_bedrooms'].apply(lambda x: train_median_bedroom if pd.isna(x) else x)
X_test['property_bedrooms_missing'] = X_test['property_bedrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
X_test['property_bedrooms'] = X_test['property_bedrooms'].apply(lambda x: train_median_bedroom if pd.isna(x) else x)

# Encode bathrooms
bathroom_imputer = SingleImputer(strategy={'property_bathrooms':'least squares'}, predictors=['property_bedrooms', 'property_beds', 'property_max_guests'])
bathroom_imputer.fit(X_train)
#X_train['property_bathrooms_missing'] = X_train['y'].apply(lambda x: 0 if not pd.isna(x) else 1)
#X_valid['property_bathrooms_missing'] = X_valid['property_bathrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
#X_test['property_bathrooms_missing'] = X_test['property_bathrooms'].apply(lambda x: 0 if not pd.isna(x) else 1)
#X_train = bathroom_imputer.transform(X_train)
#X_valid = bathroom_imputer.transform(X_valid)
#X_test = bathroom_imputer.transform(X_test)

# Encode property_beds
X_train['property_beds'] = X_train['property_beds'].apply(lambda x: 1 if not pd.isna(x) else x)
X_valid['property_beds'] = X_valid['property_beds'].apply(lambda x: 1 if not pd.isna(x) else x)
X_test['property_beds'] = X_test['property_beds'].apply(lambda x: 1 if not pd.isna(x) else x)

# with bathrooms least squares: 45.598270459878535
# with bathrooms median: 45.598270459878535
# without: 45.55039312278264

# Capture difference between "reviews_first" and "reviews_last" in new features "number of days between" and remove both features
def days_between(a, b):
    if not pd.isna(a):
        d1 = datetime.datetime.strptime(str(a), "%Y-%m-%d")
        d2 = datetime.datetime.strptime(str(b), "%Y-%m-%d")
        return abs((d2-d1).days)
    else:
        return -1

X_train['first_last_daydiff'] = X_train.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)
X_valid['first_last_daydiff'] = X_valid.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)
X_test['first_last_daydiff'] = X_test.apply(lambda x: days_between(x['reviews_first'], x['reviews_last']), axis=1)

#X_train['first_last_daydiff'] = standardize(X_train['first_last_daydiff'], X_train['first_last_daydiff'])
#X_valid['first_last_daydiff'] = standardize(X_valid['first_last_daydiff'], X_train['first_last_daydiff'])
#X_test['first_last_daydiff'] = standardize(X_test['first_last_daydiff'], X_train['first_last_daydiff'])
# without standardization: 45.749259613471736
# with standardization: 45.83292936772679

# Bed type
response_map_bed_type = {"Real Bed": 2, "Pull-out Sofa": 0, "Futon": 1, "Couch": 0, "Airbed": 1}
X_train["property_bed_type"] = X_train['property_bed_type'].replace(response_map_bed_type)
X_train["property_bed_type"].fillna(0, inplace=True)

X_valid["property_bed_type"] = X_valid['property_bed_type'].replace(response_map_bed_type)
X_valid["property_bed_type"].fillna(0, inplace=True)

X_test["property_bed_type"] = X_test['property_bed_type'].replace(response_map_bed_type)
X_test["property_bed_type"].fillna(0, inplace=True)

# property_room_type
response_map_room_type = {"Entire home/apt": 2, "Private room": 1, "Shared room": 0}
X_train["property_room_type"] = X_train['property_bed_type'].replace(response_map_room_type)
X_train["property_room_type"].fillna(0, inplace=True)

X_valid["property_room_type"] = X_valid['property_bed_type'].replace(response_map_room_type)
X_valid["property_room_type"].fillna(0, inplace=True)

X_test["property_room_type"] = X_test['property_bed_type'].replace(response_map_room_type)
X_test["property_room_type"].fillna(0, inplace=True)

# Encode "property_amenities"
# Include top N most frequently occuring amenities as dummy-columns
one_hot_amenities, amen_topN_list = binarize_amenities(X_train['property_amenities'], 30, True)
X_train = pd.concat([X_train, one_hot_amenities], axis=1)

one_hot_amenities_valid = binarize_amenities(X_valid['property_amenities'], 10, False)
one_hot_amenities_valid.drop(one_hot_amenities_valid.columns.difference(amen_topN_list), axis=1, inplace=True)
X_valid = pd.concat([X_valid, one_hot_amenities_valid], axis=1)

one_hot_amenities_test = binarize_amenities(X_test['property_amenities'], 10, False)
one_hot_amenities_test.drop(one_hot_amenities_test.columns.difference(amen_topN_list), axis=1, inplace=True)
X_test = pd.concat([X_test, one_hot_amenities_test], axis=1)

# Encode "host_verified"
# Include top N most frequently occuring amenities as dummy-columns
one_hot_verified, verified_topN_list = binarize_amenities(X_train['host_verified'], 10, True)
X_train = pd.concat([X_train, one_hot_verified], axis=1)

one_hot_verified_valid = binarize_amenities(X_valid['host_verified'], 10, False)
one_hot_verified_valid.drop(one_hot_verified_valid.columns.difference(verified_topN_list), axis=1, inplace=True)
X_valid = pd.concat([X_valid, one_hot_verified_valid], axis=1)

one_hot_verified_test = binarize_amenities(X_test['host_verified'], 10, False)
one_hot_verified_test.drop(one_hot_verified_test.columns.difference(verified_topN_list), axis=1, inplace=True)
X_test = pd.concat([X_test, one_hot_verified_test], axis=1)

# One-Hot encode "host_verified"
X_train['nr_verifications'] = verified_nr(X_train['host_verified'])
X_valid['nr_verifications'] = verified_nr(X_valid['host_verified'])
X_test['nr_verifications'] = verified_nr(X_test['host_verified'])

# Include "host_response_time" as ordinal variable
response_map_host_response_rate = {"within an hour": 1, "within a few hours": 2, "within a day": 3, "a few days or more": 4}
X_train["response_time"] = X_train['host_response_time'].replace(response_map_host_response_rate)
X_train["response_time"].fillna(5, inplace=True)
X_valid["response_time"] = X_valid['host_response_time'].replace(response_map_host_response_rate)
X_valid["response_time"].fillna(5, inplace=True)
X_test["response_time"] = X_test['host_response_time'].replace(response_map_host_response_rate)
X_test["response_time"].fillna(5, inplace=True)

# Encode property_sqfeet
thresh_sqf = 150
X_train['property_sqfeet'] = X_train['property_sqfeet'].fillna(0)
X_train['sqfeet_missing'] = X_train['property_sqfeet'].apply(lambda x: 1 if x == 0 else 0)
X_train['property_sqfeet'] = X_train['property_sqfeet'].apply(lambda x: x*10.764 if x <= thresh_sqf else x)

X_valid['property_sqfeet'] = X_valid['property_sqfeet'].fillna(0)
X_valid['sqfeet_missing'] = X_valid['property_sqfeet'].apply(lambda x: 1 if x == 0 else 0)
X_valid['property_sqfeet'] = X_valid['property_sqfeet'].apply(lambda x: x*10.764 if x <= thresh_sqf else x)

X_test['property_sqfeet'] = X_test['property_sqfeet'].fillna(0)
X_test['sqfeet_missing'] = X_test['property_sqfeet'].apply(lambda x: 1 if x == 0 else 0)
X_test['property_sqfeet'] = X_test['property_sqfeet'].apply(lambda x: x*10.764 if x <= thresh_sqf else x)

# Encode reviews_value
impute_reviews_value = SingleImputer(strategy={'reviews_value':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_rating',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_location'
                                                                                            ])

impute_reviews_value.fit(X_train)
X_train = impute_reviews_value.transform(X_train)
X_valid = impute_reviews_value.transform(X_valid)
X_test = impute_reviews_value.transform(X_test)

# Encode reviews_location
impute_reviews_location = SingleImputer(strategy={'reviews_location':'least squares'}, predictors=['host_nr_listings',
                                                                                         'reviews_num',
                                                                                         'booking_availability_365',
                                                                                         'reviews_rating',
                                                                                         'reviews_cleanliness',
                                                                                         'reviews_checkin',
                                                                                         'reviews_value'
                                                                                            ])

impute_reviews_location.fit(X_train)
X_train = impute_reviews_location.transform(X_train)
X_valid = impute_reviews_location.transform(X_valid)
X_test = impute_reviews_location.transform(X_test)

# Encode "extra"
# Include top N most frequently occuring 'extra"-values' as dummy-columns
one_hot_extra, extra_topN_list = binarize_amenities(X_train['extra'], 5, True)
X_train = pd.concat([X_train, one_hot_extra], axis=1)

one_hot_extra_valid = binarize_amenities(X_valid['extra'], 10, False)
one_hot_extra_valid.drop(one_hot_extra_valid.columns.difference(extra_topN_list), axis=1, inplace=True)
X_valid = pd.concat([X_valid, one_hot_extra_valid], axis=1)

one_hot_extra_test = binarize_amenities(X_test['extra'], 10, False)
one_hot_extra_test.drop(one_hot_extra_test.columns.difference(extra_topN_list), axis=1, inplace=True)
X_test = pd.concat([X_test, one_hot_extra_test], axis=1)

# Encode booking_cancel_policy
cancel_map = {'flexible': 10000, 'moderate': 11000, 'strict': 11100, 'super_strict_30': 11110}
X_train['booking_cancel_policy'] = X_train['booking_cancel_policy'].replace(cancel_map)
X_valid['booking_cancel_policy'] = X_valid['booking_cancel_policy'].replace(cancel_map)
X_test['booking_cancel_policy'] = X_test['booking_cancel_policy'].replace(cancel_map)


# Anomaly detection with IsolationForest
clf = IsolationForest(random_state=0, contamination=0.04).fit(X_train[['property_lat',
                               'property_lon',
                               'property_max_guests',
                               'booking_min_nights',
                               'first_last_daydiff',
                               #'reviews_acc',
                               'booking_price_covers',
                               'property_last_updated',
                               'target'
                               #'property_sqfeet'
                                ]])
outliers = pd.DataFrame(clf.predict(X_train[['property_lat',
                               'property_lon',
                               'property_max_guests',
                               'booking_min_nights',
                               'first_last_daydiff',
                               #'reviews_acc',
                               'booking_price_covers',
                               'property_last_updated',
                               'target'
                               #'property_sqfeet'
                                ]]))
#print("outliers: ", outliers[outliers < 1])
X_train = pd.concat([X_train, outliers], axis=1)
X_train.rename(columns={0: 'outlier'}, inplace=True)
#print(X_train[['property_sqfeet', 'outlier']][X_train['property_sqfeet'] > 0])
print('before: ', len(X_train))
outlier_set = X_train[['property_lat',
                               'property_lon',
                               'property_max_guests',
                               'booking_min_nights',
                               'first_last_daydiff',
                               'booking_price_covers',
                               'property_last_updated',
                               'target']][X_train['outlier'] < 1]
#plt.scatter(outlier_set['property_lat'], outlier_set['property_lon'])
#plt.show()
X_train = X_train[X_train['outlier'] == 1]
y_train = X_train['target']
print('after: ', len(X_train))

#print(len(X_train[X_train['booking_price_covers'] >4]))

# Thin out dataset with Isolation Forest
#before IsolationForest:  5196
#after IsolationForest:  4988

# contamination=0.01 => RMSE: 45.64449902470632 & corr: 0.00213
# contamination=0.05 => RMSE: 45.571575735533735 & corr: -0.002918
# contamination=0.02 => RMSE: 45.634763281134134 & corr: -0.014354
# contamination=0.03 => RMSE: 45.83638134864261 & corr: 0.021197
# contamination=0.04 => RMSE: 45.4762461418208 & corr: -0.015524


# Prediction
# Data Columns to keep for prediction
X_train.drop(['target'], axis=1, inplace=True)
X_valid.drop(['target'], axis=1, inplace=True)
keep_columns_for_prediction = ['property_lat',
                               'property_lon',
                               'property_zipcode_freq',
                               'property_zipcode_targetmean',# does not have target-corr >1%
                               #'property_feature_type0',
                               'property_feature_type1',
                               #'property_feature_type2',
                               'property_feature_type3',
                               #'property_bed_type',
                               'first_last_daydiff',
                               'reviews_acc',
                               'booking_price_covers',
                               #'property_room_type',
                               'property_last_updated',
                               'cluster_id', # with: 45.48, without: 45.52
                               #'property_bathrooms',
                               #'property_bathrooms_missing'
                               #'response_time',
                               #'nr_verifications'
                               #'city',
                               #'WirelessInternet',
                               #'Heating',
                               #'Essentials',
                               #'Smokedetector',
                               #'Washer',
                               #'Shampoo',
                               #'Internet',
                               'Hangers',
                               'Familykidfriendly',
                               #'TV',
                               #'Hairdryer',
                               #'Laptopfriendlyworkspace',
                               #'Iron',
                               #'Buzzerwirelessintercom',
                               #'Fireextinguisher',
                               #'Elevatorinbuilding',
                               #'Dryer',
                               #'CableTV',
                               #'translationmissing:en.hosting_amenity_50',
                               #'Firstaidkit',
                               '24-hourcheck-in',
                               #'translationmissing:en.hosting_amenity_49',
                               #'Carbonmonoxidedetector',
                               #'Safetycard',
                               #'Smokingallowed',
                               #'Freeparkingonpremises',
                               #'Lockonbedroomdoor',
                               #'Breakfast',
                               #'Petsallowed',
                               'property_sqfeet',
                               #'property_beds',
                               'booking_availability_365', # doesnt have target-corr >1% in here it increases RMSE, in test it doesn't
                               #'booking_availability_30',
                               #'reviews_value',
                               #'reviews_location',
                               'HostHasProfilePic', #does not have a target-correlation >1%
                               'HostIdentityVerified',
                               #'InstantBookable',
                               'booking_cancel_policy',
                               #'phone',
                               'reviews_rating',# does not have target-corr >1%
                               'reviews_cleanliness',
                               'reviews_location',
                               'reviews_value'
                               ]

keep_columns_for_prediction1 = ['reviews_acc', 'reviews_location', 'cluster_id', 'property_feature_type1', 'property_feature_type3', 'first_last_daydiff',
                                '24-hourcheck-in', 'Familykidfriendly', 'Hangers', 'HostIdentityVerified']
# with all:
# without rev_communication: 45.493859556653824
#with verification and response_time: 45.57017186907026
#without verification and response_time: 45.61486905009188
#with verification and without response_time: 45.598270459878535
#without verification and with response_time: 45.55039312278264


X_train.drop(X_train.columns.difference(keep_columns_for_prediction), axis=1, inplace=True)
X_valid.drop(X_valid.columns.difference(keep_columns_for_prediction), axis=1, inplace=True)
x_test_property_id = X_test['property_id']
X_test.drop(X_test.columns.difference(keep_columns_for_prediction), axis=1, inplace=True)
X_train['cluster_id'] = X_train['cluster_id'].fillna(-1)

# Model XGBOOST
xgb = xgboost.XGBRegressor(eta=0.02, max_depth=3, subsample=0.8, min_child_weight=5)
xgb.fit(X_train, y_train)
plot_importance(xgb)
#plt.show()
xgb_train = xgb.predict(X_train)
xgb_valid = xgb.predict(X_valid)
xgb_test = xgb.predict(X_test)

print('Mean-Pred- RMSE: ', rmse(y_train.mean(), y_valid, False))
print("Mean-Pred. - MeanAbsoluteError: ", meanAbsoluteError(y_train.mean(), y_valid, False))
print('\n')
print('xgb_valid - RMSE: ', rmse(xgb_valid, y_valid, False))
print("xgb_valid - MeanAbsoluteError: ", meanAbsoluteError(xgb_valid, y_valid, False))

# Model RANDOM FOREST
randomForest_model = sklearn.ensemble.RandomForestRegressor(n_estimators=400, max_depth=3, bootstrap=True, max_features=0.8)
randomForest_model.fit(X_train.values, y_train.values)
randomForest_train = randomForest_model.predict(X_train.values)
randomForest_valid = randomForest_model.predict(X_valid.values)
print('\n')
print('randomForest_valid - RMSE: ', rmse(randomForest_valid, y_valid, False))
print("randomForest_valid - MeanAbsoluteError: ", meanAbsoluteError(randomForest_valid, y_valid, False))

# Model LINEAR REGRESSION
linear_model = sklearn.linear_model.Ridge(alpha=1)
linear_model.fit(X_train.values, y_train.values)
linear_train = linear_model.predict(X_train.values)
linear_valid = linear_model.predict(X_valid.values)
print('\n')
print('linear_valid - RMSE: ', rmse(linear_valid, y_valid, False))
print("linear_valid - MeanAbsoluteError: ", meanAbsoluteError(linear_valid, y_valid, False))


pd.DataFrame(pd.concat([pd.DataFrame(x_test_property_id), pd.DataFrame(xgb_test)], axis=1)).to_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/modelPREDS/xgb_test_predictions.csv', index=False)


given_mean = y_train.mean()
mean_set = pd.DataFrame(x_test_property_id)
mean_set['PRED'] = [given_mean for i in range(len(xgb_test))]

# Save predictions in a new csv file for upload
#mean_set.to_csv('/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment1/data/modelPREDS/mean_predictor_set.csv', index=False)

#plt.plot(range(len(y_valid)), y_valid)
#plt.plot(range(len(y_valid)), xgb_valid)
print(pd.concat([X_valid, pd.DataFrame(y_valid)], axis=1).corr())
#msno.dendrogram(pd.concat([X_valid, pd.DataFrame(y_valid)], axis=1))
#plt.show()
