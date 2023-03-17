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
import seaborn as sb
from dtreeviz.trees import *



pd.set_option('display.max_rows', 7000)
pd.set_option('display.max_columns', 7000)
pd.set_option('display.width', 7000)

defects = pd.DataFrame(pd.read_csv('/Users/ivoarasin/Desktop/Master/Semester Two/Data Science for Business/Data Science (Verbeke)/assignment/Assignment 1 - Defects dataset.csv'))
print(defects.columns)

#defect_cars = yp.ProfileReport(data, title="DefectCarsProfilingReport")
#defect_cars.to_file("DefectCarsProfilingReport.html")

#ax1 = data.groupby(['defect'])['plant'].value_counts().unstack().plot(kind='bar',stacked = True)
pal = sb.color_palette("rocket", len([0,1,2,3]))

defects_hist_plot = sb.histplot(binwidth=0.5, x='plant', hue='defect', data=defects, stat="count", multiple="stack", palette=pal)
defects_hist_plot.set(xlabel='Defects', ylabel='Count', title='Type of defects per plant')
#plt.show()
target_variable = defects['defect']
defects.drop(['defect'], axis=1, inplace=True)

X = defects
map_quali_worker = {"vhigh": 4, "high": 3, "med": 2, "low": 1}
X["quality_worker"] = X["quality worker experience"].replace(map_quali_worker)
defects.drop(['quality worker experience'], axis=1, inplace=True)

map_custom_options = {"more": 6, "4": 4, "med": 2, "2": 2}
X["custom_options"] = X["number custom options"].replace(map_quali_worker)
defects.drop(['number custom options'], axis=1, inplace=True)
X = pd.get_dummies(X)

train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(X, target_variable, test_size=0.3)

defect_tree = sklearn.tree.DecisionTreeClassifier()
defect_tree.fit(train_data, train_target)
defect_tree_predictions = defect_tree.predict(test_data)
dtreeviz.model(defect_tree, X_train=train_data, y_train=train_target, target_name="defects",feature_names=train_data.columns, class_names=['a', 'b', 'c', 'd'])





#print(sklearn.metrics.confusion_matrix(defect_tree_predictions, test_target))



