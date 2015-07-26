#!/usr/bin/python

'''Set of helper functions for the processing of the enron email + financial (E + F) data set'''
import pickle
import sys
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from sklearn.feature_selection import SelectKBest,SelectPercentile

#remove a key for a dictionary object where keys is a list
def remove_outliers(dictionary, outliers):
    """ this function removes a list of keys from a dictionary object """
    for outlier in outliers:
        dictionary.pop(outlier, 0)


#function to add features (expandable to add multiple features)

def add_poi_interaction_ratio(dictionary):
    """
    Adds poi email interaction ratio data dictionary
    """
    for person in dictionary:

        # poi_ratio: Compute and add ratio of messages involving (to, from, or shared) POI to total messages
        try:
            total_messages = dictionary[person]['from_messages'] + dictionary[person]['to_messages']
            from_poi = dictionary[person]["from_poi_to_this_person"]
            to_poi =  dictionary[person]["from_this_person_to_poi"]
            shared_poi = dictionary[person]["shared_receipt_with_poi"]
            poi_related_messages = from_poi +\
                                    to_poi +\
                                    shared_poi
            #convert data types to float
            total_messages = float(total_messages)
            from_poi = float(from_poi)
            to_poi = float(to_poi)
            shared_poi = float(shared_poi)

            poi_ratio = poi_related_messages / total_messages
            dictionary[person]['poi_ratio'] = poi_ratio

        except:
            dictionary[person]['poi_ratio'] = 'NaN'


def add_poi_from_ratio(dictionary):
    """
    Adds ratio of emails from pois to the data dictionary
    """
    for person in dictionary:

        # poi_from_ratio: Compute and add ratio of messages from POI to total messages
        try:
            total_messages = dictionary[person]['from_messages'] + dictionary[person]['to_messages']
            from_poi = dictionary[person]["from_poi_to_this_person"]
            
           

            poi_from_ratio = float(from_poi) / float(total_messages)
            dictionary[person]['poi_from_ratio'] = poi_from_ratio
            
            

        except:
            dictionary[person]['poi_from_ratio'] = 'NaN'

def add_poi_to_ratio(dictionary):
    """
    Adds ratio of emails to pois to the data dictionary
    """
    for person in dictionary:

        # poi_to_ratio: Compute and add ratio of messages to POI to total messages
        try:
            total_messages = dictionary[person]['from_messages'] + dictionary[person]['to_messages']
            to_poi =  dictionary[person]["from_this_person_to_poi"]
            



            poi_to_ratio = float(to_poi) / float(total_messages)
            dictionary[person]['poi_to_ratio'] = poi_to_ratio
            
            
            

        except:
            dictionary[person]['poi_to_ratio'] = 'NaN'



    return dictionary


add_feature_function_list = [add_poi_to_ratio,add_poi_from_ratio,add_poi_interaction_ratio]

def add_features(function_list, dictionary):
    '''
    Adds features to the data dictionary by iterating a list of functions which add features
    '''
    for function in function_list:
        function(dictionary) 





def get_features(dictionary):
    '''
    Accepts the enron data dictionary and returns features
    '''
    feature_keys = dictionary['SKILLING JEFFREY K'].keys()
    return list(feature_keys)

'''
def get_k_best(dictionary, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection returning:
    {feature:score}
    """
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features
'''

def get_nan_counts(dictionary):
    '''
    converts 'NaN' string to np.nan returning a pandas
    dataframe of each feature and it's corresponding
    percent null values (nan)
    '''
    my_df = pd.DataFrame(dictionary).transpose()
    nan_counts_dict = {}
    for column in my_df.columns:
        my_df[column] = my_df[column].replace('NaN',np.nan)
        nan_counts = my_df[column].isnull().sum()
        nan_counts_dict[column] = round(float(nan_counts)/float(len(my_df[column])) * 100,1)
    df = pd.DataFrame(nan_counts_dict,index = ['percent_nan']).transpose()
    df.reset_index(level=0,inplace=True)
    df = df.rename(columns = {'index':'feature'})
    return df


def get_k_best(dictionary, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection returning:
    {feature:score}
    """
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    pairs = zip(features_list[1:], scores)
    #combined scores and features into a pandas dataframe then sort 
    k_best_features = pd.DataFrame(pairs,columns = ['feature','score'])
    k_best_features = k_best_features.sort('score',ascending = False)
    
    
    #merge with null counts    
    df_nan_counts = get_nan_counts(dictionary)
    k_best_features = pd.merge(k_best_features,df_nan_counts,on= 'feature')  
    
    #eliminate infinite values
    k_best_features = k_best_features[np.isinf(k_best_features.score)==False]
    print 'Feature Selection by k_best_features\n'
    print "{0} best features in descending order: {1}\n".format(k, k_best_features.feature.values[:k])
    print '{0}\n'.format(k_best_features[:k])
    
    
    return k_best_features[:k]


def extract_data():
    '''
    
    
    '''
    
    
    return



'''
This isn't quite working as intended yet.  Decided to circle back later in lieu of k best and PCA.
I'm keeping the code here for future work.

def get_k_percentile(dictionary, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection returning:
    {feature:score}
    """
    data = featureFormat(dictionary, features_list)
    labels, features = targetFeatureSplit(data)

    k_percentile = SelectPercentile(percentile=k)
    k_percentile.fit_transform(features, labels)
    scores = k_percentile.scores_
    pairs = zip(features_list[1:], scores)
    #combined scores and features into a pandas dataframe then sort 
    k_percentile_features = pd.DataFrame(pairs,columns = ['feature','score'])
    k_percentile_features = k_percentile_features.sort('score',ascending = False)
    
    
    #merge with null counts    
    df_nan_counts = get_nan_counts(dictionary)
    k_percentile_features = pd.merge(k_percentile_features,df_nan_counts,on= 'feature')  
    
    #eliminate infinite values
    k_percentile_features = k_percentile_features[np.isinf(k_percentile_features.score)==False]
    print "{0}th percentile features in descending order: {1}\n".format(100-k, k_percentile_features.feature.values)

    
    return k_percentile.fit(features,labels)
    '''

    


