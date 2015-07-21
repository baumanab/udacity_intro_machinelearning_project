#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

#additional imports
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
import enron_tools
import enron_evaluate


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
data_label = 'poi'
features_list = enron_tools.get_features(data_dict)

#email address does not help with prediction and causes exeception, remove
features_list.remove('email_address')

#other is not a well defined feature, remove
features_list.remove('other')

#remove the data_label so that it can be re-added as the first feature element
features_list.remove('poi')

#reassemble feaures with the data label as the first element
features_list = [data_label] + features_list


### Task 2: Remove outliers

outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']

enron_tools.remove_outliers(data_dict, outliers)

### Task 3: Create new feature(s)

#create list of functions for use as argument to add_features function
add_feature_function_list = [enron_tools.add_poi_to_ratio,enron_tools.add_poi_from_ratio,enron_tools.add_poi_interaction_ratio]
#add features to data_dict
enron_tools.add_features(add_feature_function_list,data_dict)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)