#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
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
def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
        
    ### your code goes here!

    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    clf = SVC(kernel="rbf", C = 10000.0)
    t0 = time()
    #features_train = features_train[:len(features_train)/100] 
    #xlabels_train = labels_train[:len(labels_train)/100] 
    clf.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"
    t0 = time()
    pred = clf.predict(features_test) 
    n = []
    [n.append(e) for e in pred if e == 1]


    chris = len(n)

    print "prediction time:", round(time()-t0, 3), "s"
    accuracy =  accuracy_score(pred,labels_test)
    return accuracy, chris 

acc, chris = classify(features_train,labels_train)
print acc, chris
#########################################################