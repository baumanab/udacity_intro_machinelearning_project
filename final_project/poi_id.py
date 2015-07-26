# coding: utf-8


#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

#additional imports
import numpy as np
from sklearn.feature_selection import SelectKBest
import enron_tools
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier

sep = '##############################################################################################'
sep2 = '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### create list of functions for use as argument to add_features function
add_feature_function_list = [enron_tools.add_poi_to_ratio,enron_tools.add_poi_from_ratio,enron_tools.add_poi_interaction_ratio]

## add features to data_dict
enron_tools.add_features(add_feature_function_list,data_dict)

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

data_label = 'poi'
features_list = enron_tools.get_features(data_dict)


## email address does not help with prediction and causes exeception, remove
features_list.remove('email_address')


##remove the data_label so that it can be re-added as the first feature element
features_list.remove('poi')

##reassemble feaures with the data label as the first element
features_list = [data_label] + features_list

############################################################################################################

### Task 2: Remove outliers

outliers = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']

enron_tools.remove_outliers(data_dict, outliers)


### Store to my_dataset for easy export below.
my_dataset = data_dict

###############################################################################################

### Continue Feature Selection and dimensionality reduction via get_k_best

## get k (k represents number of features) best features
k = 10
k_best_features = enron_tools.get_k_best(data_dict,features_list,k)

print sep

# assemble feature list
my_features_list = [data_label] + list(k_best_features.feature.values)



###################################################################################################


### extract features and labels for gridsearch optimization

# data extraction using k_best features list
data = featureFormat(my_dataset, my_features_list, sort_keys = True)

tru, trn = targetFeatureSplit(data)

## scale extracted features
scaler = preprocessing.MinMaxScaler()
trn = scaler.fit_transform(trn)


# Set up cross validator (will be used for tuning all classifiers)
cv = cross_validation.StratifiedShuffleSplit(tru,
                                            n_iter = 10,
                                             random_state = 42)

## Evaluate Final Adaboost Classifier

# load tuned classifier pipeline


best_a_pipe = pickle.load(open('best_clf_pipe.pkl', "r") )




print 'best_a_clf\n'
best_a_pipe
test_classifier(best_a_pipe,my_dataset,my_features_list)
print sep

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(best_a_pipe, my_dataset, my_features_list)
