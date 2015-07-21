#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from sklearn import tree
#clf_minsplt2 = tree.DecisionTreeClassifier(min_samples_split = 2)
clf_minsplt40 = tree.DecisionTreeClassifier(min_samples_split = 40)
#clf_minsplt2.fit(features_train, labels_train)
clf_minsplt40.fit(features_train,labels_train)
#pred2 = clf_minsplt2.predict(features_test)
pred40 = clf_minsplt40.predict(features_test)


#acc_min_samples_split_2 = accuracy_score(pred2,labels_test)

acc_min_samples_split_40 = accuracy_score(pred40,labels_test)


def submitAccuracies():
  return {"acc_min_samples_split_40":round(acc_min_samples_split_40,3)}


acc = submitAccuracies()

print acc, len(features_train[0])

#########################################################


