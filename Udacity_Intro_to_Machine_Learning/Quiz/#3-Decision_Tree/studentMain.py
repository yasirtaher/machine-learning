#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys

sys.path.append("../common_files/")
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify
from sklearn import tree

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!

clf = classify(features_train, labels_train)
acc = clf.score(features_test, labels_test)


def submitAccuracies():
    return {"acc": round(acc, 3)}


print submitAccuracies()

#### grader code, do not modify below this line
prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())

### #### #### ### #### #### ### #### ####

### Quiz: Decision Tree Accuracy
clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf50 = tree.DecisionTreeClassifier(min_samples_split=50)

clf2.fit(features_train, labels_train)
clf50.fit(features_train, labels_train)

acc2 = clf2.score(features_test, labels_test)
acc50 = clf50.score(features_test, labels_test)


# method 2
# from sklearn.metrics import accuracy_score
# pred = clf.predict(features_test)
# acc = accuracy_score(pred, labels_test)


def submitDTAccuracies():
    return {"acc_min_samples_split_2": round(acc2, 3),
            "acc_min_samples_split_50": round(acc50, 3)}


print submitDTAccuracies()
#### end of accuracy

### #### ####### #### ####### #### ####### #### ####

# Entropy Calculation
