
# coding: utf-8

# In[1]:

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
import enron_evaluate
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn import preprocessing
from sklearn import cross_validation
import enron_evaluate

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

## other is not a well defined feature, remove
features_list.remove('other')

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

### Continue Feature Selection and dimensionality reduction via get_k_best and pca

## get k (k represents number of features) best features
k = 10
k_best_features = enron_tools.get_k_best(data_dict,features_list,10)

print sep

# assemble feature list
my_features_list = [data_label] + list(k_best_features.feature.values)

## pca

'''
pca features/data can be scaled or standardized, I experimented with both and
ultimately opted to go with feature scaling.  Below is the code for stanardizing

    std = preprocessing.StandardScaler()
    std_pca_data = preprocessing.StandardScaler().fit_transform(data_for_pca)
'''

# remove label from features_list
features_for_pca = features_list[1:]

# extract features
data_for_pca = featureFormat(my_dataset, features_for_pca, sort_keys = True)

# scale features
scale_pca_data = preprocessing.MinMaxScaler().fit_transform(data_for_pca)

# set up PCA to explain pre-selected % of variance (perc_var)
perc_var = .95
pca = PCA(n_components=perc_var)

# fit and transform
pca_transform = pca.fit_transform(scale_pca_data)

# Starting features and ending components
num_features = len(features_for_pca)
components = pca_transform.shape[1]
print 'PCA\n'
print 'Explained Variance: {0}\n Original Number of Dimensions: {1}\n Final Dimensions: {2}\n'.format(perc_var,num_features,components)
print sep

###################################################################################################

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## Gaussian Classifier
from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.

### Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
a_clf = AdaBoostClassifier(algorithm= 'SAMME')


### Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()


## Evaluate Initial Classifiers using k_best features

print 'Evaluate Initial Classifiers using k_best features\n'
kbest_classifiers_list = [g_clf,a_clf,dt_clf]

print 'Local Evaluator\n'
enron_evaluate.evaluate_validate(kbest_classifiers_list,my_dataset,my_features_list,scale_features=True)

print sep

print 'tester.py evaluator\n'
test_classifier(g_clf,my_dataset,my_features_list, scale_features = True)
print sep2
test_classifier(a_clf,my_dataset,my_features_list, scale_features = True)
print sep2
test_classifier(dt_clf,my_dataset,my_features_list, scale_features = True)
print sep


## Evaluate Initial Classifiers using PCA
## Note that feature selection is the only way to "tune" GaussianNB

print 'Evaluate Initial Classifiers using PCA\n'
g_pipe = Pipeline(steps=[('pca', pca), ('gaussian', g_clf)])
a_pipe = Pipeline(steps=[('pca', pca), ('adaboost', a_clf)])
dt_pipe = Pipeline(steps = [('pca',pca),('decision_tree', dt_clf)])

pca_classifiers_list = [g_pipe,a_pipe,dt_pipe]

print 'Local Evaluator\n'
enron_evaluate.evaluate_validate(pca_classifiers_list,my_dataset,features_list,scale_features=True)

print sep

print 'tester.py evaluator\n'
test_classifier(g_pipe,my_dataset,features_list, scale_features = True)
print sep2
test_classifier(a_pipe,my_dataset,features_list, scale_features = True)
print sep2
test_classifier(dt_pipe,my_dataset,features_list, scale_features = True)

print sep



###################################################################################################

### extract features and labels for gridsearch optimization

data = featureFormat(my_dataset, my_features_list, sort_keys = True)
tru, trn = targetFeatureSplit(data)

## scale extracted features
scaler = preprocessing.MinMaxScaler()
trn = scaler.fit_transform(trn)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print "Tune Classifiers\n"

## Tune decision tree via gridsearch

# Set up cross validator (will be used for tuning all classifiers)
cv = cross_validation.StratifiedShuffleSplit(tru,
                                            n_iter = 10,
                                             random_state = 42)
# set up estimator and pipeline, using PCA for feature selection
estimators = [('reduce_dim', PCA()),('dec_tree',dt_clf)]
dtclf = Pipeline(estimators)

# set up paramaters dictionary
dt_params = dict(reduce_dim__n_components=[.95],
              dec_tree__criterion=("gini","entropy"),
                  dec_tree__min_samples_split=[1,2,4,8,16,32],
                   dec_tree__min_samples_leaf=[1,2,4,8,16,32],
                   dec_tree__max_depth=[None,1,2,4,8,16,32])

# set up gridsearch
dt_grid_search = GridSearchCV(dtclf, param_grid = dt_params,
                          scoring = 'f1', cv =cv)

# pass data into into the gridsearch via fit
dt_grid_search.fit(trn,tru)

print 'Decision tree tuning\n Steps: {0}\n, Best Parameters: {1}\n '.format(dtclf.steps,dt_grid_search.best_params_,dt_grid_search.best_score_)
print sep2
# pick a winner
best_dtclf = dt_grid_search.best_estimator_

## Tune adaboost via gridsearch

# set up estimator and pipeline, using PCA for feature selection
estimators = [('reduce_dim', PCA()),('adaboost',a_clf)]
aclf = Pipeline(estimators)

# set up paramaters dictionary
a_params = dict(reduce_dim__n_components=[.95],
              adaboost__n_estimators=[5, 10, 30, 40, 50, 100,150],
                  adaboost__learning_rate=[0.1, 0.5, 1, 1.5, 2, 2.5],
                   adaboost__algorithm=('SAMME', 'SAMME.R'))

# set up gridsearch
a_grid_search = GridSearchCV(aclf, param_grid = a_params,
                          scoring = 'f1', cv =cv)
# pass data into into the gridsearch via fit
a_grid_search.fit(trn,tru)

print 'Adaboost tuning\n Steps: {0}\n, Best Parameters: {1}\n '.format(aclf.steps,a_grid_search.best_params_,a_grid_search.best_score_)
print sep2
# pick a winner
best_aclf = a_grid_search.best_estimator_


## Tune adaboost with best decision tree, via gridsearch 

# Assign the best parameters from decision tree tuning to a variable (cut and paste for now 
# there has to be a better way to do this)

best_dt_params = DecisionTreeClassifier(compute_importances=None, criterion='entropy',
            max_depth=16, max_features=None, max_leaf_nodes=None,
            min_density=None, min_samples_leaf=1, min_samples_split=8,
            random_state=None, splitter='best')

# Set up classifier
adt_clf = AdaBoostClassifier(best_dt_params)

# Set up estimator and pipeline, using PCA for dimensitonality reduction
estimators = [('reduce_dim', PCA()),('adaboost',adt_clf)]
adtclf = Pipeline(estimators)

# Set up parameters dictionary
adt_params = dict(reduce_dim__n_components=[.95],
              adaboost__n_estimators=[5, 10, 30, 40, 50, 100,150,200],
                  adaboost__learning_rate=[0.1, 0.5, 1, 1.5, 2, 2.5],
                   adaboost__algorithm=('SAMME', 'SAMME.R'))

# Set up grid search
adt_grid_search = GridSearchCV(adtclf, param_grid = adt_params,
                          scoring = 'f1', cv = cv)

# Pass data into the gridsearch by calling fit
adt_grid_search.fit(trn,tru)

print 'Adaboost with Tuned Decision Tree, tuning\n Steps: {0}\n, Best Parameters: {1}\n '.format(adtclf.steps,adt_grid_search.best_params_,adt_grid_search.best_score_)
print sep2
# pick a winner
best_adtclf = adt_grid_search.best_estimator_

## Evaluate Tuned Classifiers

print 'Evaluate Tuned Classifiers\n'

adt_pipe = Pipeline(steps=[('pca',pca),('adaboost_dt',best_adtclf)])

dt_pipe = Pipeline(steps=[('pca',pca),('dt',best_dtclf)])

best_a_pipe = Pipeline(steps=[('pca',pca),('adaboost',best_aclf)])


print 'best_dt_clf\n'
test_classifier(dt_pipe,my_dataset,my_features_list, scale_features = True, std_features= False)
print sep2

print 'best_a_clf\n'
test_classifier(best_a_pipe,my_dataset,my_features_list, scale_features = True, std_features= False)
print sep2

print 'best_adt_clf\n'
test_classifier(adt_pipe,my_dataset,my_features_list, scale_features = True, std_features= False)
print sep

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(best_aclf, my_dataset, features_list)

