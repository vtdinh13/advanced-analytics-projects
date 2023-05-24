import csv
import json
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
import json
import networkx as nx

path_graphml = '/Users/ivoarasin/Desktop/Master/Semester Two/AdvAnalyticsinBus/pythonProjects/Assignment4/twitch.graphml'
nx.read_graphml(path_graphml)