#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def my_svm(kernel='linear', C=1):
    # the classifier
    clf = SVC(kernel=kernel, C=C)
    # train
    t0 = time()
    clf.fit(features_train, labels_train)
    print "\ntraining time:", round(time() - t0, 3), "s"

    # predict
    t0 = time()
    pred = clf.predict(features_test)
    print "predicting time:", round(time() - t0, 3), "s"

    accuracy = accuracy_score(pred, labels_test)

    print '\naccuracy = {0}'.format(accuracy)
    return pred


# 1% of total training set
features_train = features_train[:len(features_train) / 100]
labels_train = labels_train[:len(labels_train) / 100]

# Optimize C Parameter
# for C in [10, 100, 1000, 10000]:
pred = my_svm('rbf', 10000)
# print '\n\n'

# 10 - 0.616040955631
# 100 -0.616040955631
# 1000 -0.821387940842
# 10000 -0.892491467577

# Extracting Predictions from an SVM
# for p in [10, 26, 50]:
#     print pred[p]

# predicted Chris emails
# print sum(pred)


#########################################################
