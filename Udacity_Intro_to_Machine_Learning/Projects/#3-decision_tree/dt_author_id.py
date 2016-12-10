#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)

# accuracy
acc = clf.score(features_test, labels_test)

def submitAccuracies():
    return {"accuracy": round(acc, 3)}

print submitAccuracies()
# end of accuracy

print features_train.shape


#########################################################
