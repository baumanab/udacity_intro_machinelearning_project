#!/usr/bin/python

def estimator_scorer(estimator,features_test,labels_test):
	'''
	function for use in scoring gridsearchcv for optimization
	of f1 after reaching recall and precision threshold
	'''
	pred = estimator.predict(features_test)
	precision = precision_score(labels_test,pred)
	recall = recall_score(labels_test,pred)
	f1 = f1_score(labels_test,pred)
	if precision > 0.3 and recall > 0.3:
		return f1
	else:
		return 0


def evaluate_validate(clf_list, dataset,feature_list,scale_features=True):
	'''
	Function accepts: list of classifiers (clf_list), list of features, and whether to scale features
	Function then:
	- extracts features and labels
	- scales features (optional)
	- evaluates clf
	- validates clf and paramaters
	- prints metrics

	'''
	from sklearn import preprocessing

	#extract features and labels
	from feature_format import featureFormat, targetFeatureSplit

	data = featureFormat(dataset, feature_list, sort_keys = True)
	labels, features = targetFeatureSplit(data)

	#scale features if necessary
	if scale_features == True:
		scaler = preprocessing.MinMaxScaler()
		features = scaler.fit_transform(features)

	#validation and evaluation

	from sklearn import cross_validation
	from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
	features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)

	for classifier in clf_list:
		clf = classifier
		clf.fit(features_train,labels_train)
		pred = clf.predict(features_test)
		score = clf.score(features_test,labels_test)
		poi_predicted = sum(pred)
		total = len(labels_test)
		precision = precision_score(labels_test,pred)
		recall = recall_score(labels_test,pred)
		f1 = f1_score(labels_test,pred)
		print('clf = {5}\n Accuracy:{0}\n Predicted Poi in test set:{1}\n Total Persons in test set:{2}\n Precision:{3}\n Recall:{4} \n F1 Score: {6} \n'.format(score,poi_predicted,total,precision,recall,clf,f1))
