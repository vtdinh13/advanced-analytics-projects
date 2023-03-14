import pandas as pd
import numpy as np
import sklearn.decomposition
import datetime
import os
from datetime import timedelta
from sklearn.neighbors import KNeighborsClassifier

# Change directory
current_dir = os.getcwd()
subdir = os.path.join(current_dir, "Assignment1")

date_columns = ['property_scraped_at', 'host_since', 'reviews_first', 'reviews_last']
train_df = pd.read_csv("train.csv", parse_dates=date_columns)
test_df = pd.read_csv("test.csv", parse_dates=date_columns)


def string_to_int(input_var):
    return_number = 0
    if input_var[0] == 'a':
        return_number = 1
    else:
        return_number = int(input_var.split(" ", 1)[0])
    return return_number


last_updated_text = train_df.property_last_updated.unique()
# last_updated_text = np.array_str(last_updated_text)
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

train_df['property_last_updated'] = train_df['property_last_updated'].apply(lambda x: dict(my_list)[x])
# train_df2['property_scraped_at'] = pd.to_datetime(train_df2['property_scraped_at'])
train_df['property_last_updated_dt'] = train_df['property_scraped_at'] + train_df['property_last_updated']

train_df['daypassed_since_lastreview'] = train_df['property_scraped_at'] - train_df['reviews_last']
train_df['daypassed_since_firstreview'] = train_df['property_scraped_at'] - train_df['reviews_first']



# some function for preprocessing
def medianImputer(dataset, columns):
    for i in columns:
        dataset[str(i)] = dataset[str(i)].fillna(dataset[str(i)].median())
    return dataset

# PCA transformation of property features as seen in the array below
pcas = train_df[['property_max_guests', 'property_bathrooms', 'property_bathrooms', 'property_bedrooms']]
train_df.drop(['property_max_guests', 'property_bathrooms', 'property_bathrooms', 'property_bedrooms'], axis=1, inplace=False)
medianImputer(pcas, ['property_max_guests', 'property_bathrooms', 'property_bathrooms', 'property_bedrooms'])
pcas = (pcas - pcas.mean())/np.std(pcas)
pca_model = sklearn.decomposition.PCA()
pca_model.fit(pcas)
flat_features = pd.DataFrame(pca_model.transform(pcas))
train_df['flatFeature1'] = flat_features.iloc[:,0]
train_df['flatFeature2'] = flat_features.iloc[:,1]
train_df['flatFeature3'] = flat_features.iloc[:,2]
#print(sum(pca_model.explained_variance_[0:2])/sum(pca_model.explained_variance_)) #0.91

# PCA transformation for another set of highly correlated variables as seen below
pcas2 = pd.DataFrame(train_df[['reviews_num','reviews_rating','reviews_acc','reviews_cleanliness','reviews_checkin','reviews_communication','reviews_location','reviews_value','reviews_per_month']])
train_df.drop(['reviews_num','reviews_rating','reviews_acc','reviews_cleanliness','reviews_checkin','reviews_communication','reviews_location','reviews_value','reviews_per_month'], axis=1, inplace=True)
medianImputer(pcas2, ['reviews_num','reviews_rating','reviews_acc','reviews_cleanliness','reviews_checkin','reviews_communication','reviews_location','reviews_value','reviews_per_month'])
pcas2 = (pcas2 - pcas2.mean())/np.std(pcas2)
pca2_model = sklearn.decomposition.PCA()
pca2_model.fit(pcas2)
review_features = pd.DataFrame(pca2_model.transform(pcas2))
train_df['reviews1'] = review_features.iloc[:,0]
train_df['reviews2'] = review_features.iloc[:,1]
train_df['reviews3'] = review_features.iloc[:,2]
train_df['reviews4'] = review_features.iloc[:,3]
#print(sum(pca2_model.explained_variance_[0:3])/sum(pca2_model.explained_variance_))


# function to impute missing values with based on KNN. e.g. impute zipcode based on latitude and longitude would look like this:
# KNNimpute(train_set, 'property_zipcode', ['property_lat', 'property_lon'], 3)
# the function directly replaces the NAs in the original dataframe, no need to append a new column with imputed values and delete the old one etc.
def KNNimpute(dataset, imputeValue, imputeWith, nr_of_neighbours):
    knn_set = pd.DataFrame()
    for i in imputeWith:
        knn_set[str(i)] = (dataset[str(i)] - dataset[str(i)].mean())/np.std(dataset[str(i)])
    knn_set[str(imputeValue)] = dataset[str(imputeValue)]
    knn_set_clean = knn_set[~knn_set.isna().any(axis=1)] # ~ operator negates if it is NA, thus returns only complete rows without NA
    knn_imputer = KNeighborsClassifier(n_neighbors=nr_of_neighbours)
    knn_imputer.fit(knn_set_clean.iloc[:,:-1].values, knn_set_clean.iloc[:,-1].values)
    dataset[str(imputeValue)] = knn_set.apply(lambda row: knn_imputer.predict([row.iloc[:-1]])[0] if pd.isna(row.iloc[-1]) else row.iloc[-1], axis=1)

KNNimpute(train_df, 'property_zipcode', ['property_lat', 'property_lon'], 3)

# Create bins for property_type corresponding to target_variance
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

train_df['property_feature_type'] = train_df.apply(lambda x: prop_type_bins(x['property_type']),axis=1)
train_df.drop(['property_type'], axis=1, inplace=True)

# Change "host_since" to number of days since host has been active up until today
def datediff_in_days_to_today(d):
    try:
        return (datetime.datetime.utcnow() - datetime.datetime.strptime(str(d), '%Y-%m-%d')).days
    except (ValueError, TypeError):
        return 0

train_df['host_since_in_days'] = train_df.apply(lambda x: datediff_in_days_to_today(x['host_since']) , axis=1)
train_df.drop(['host_since'], axis=1, inplace=True)

# Hot-One encode various columns
def hotOneEncode(dataset, columns):
    dummies = pd.DataFrame()
    encoder = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
    for i in columns:
        encoder_df = pd.DataFrame(encoder.fit_transform(dataset[[str(i)]]).toarray())
        names = []
        for j in range(len(encoder_df.columns)-1):
            dummies[str(i+str(j))] = encoder_df.iloc[:, j]
    return dummies
dums = hotOneEncode(train_df, ['property_feature_type', 'property_room_type', 'property_bed_type', 'property_scraped_at', 'host_response_time', 'booking_cancel_policy'])
train_df.drop(['property_feature_type', 'property_room_type', 'property_bed_type', 'property_scraped_at', 'host_response_time', 'booking_cancel_policy'], axis=1, inplace=True)
train_df = pd.concat([train_df, dums], axis=1)