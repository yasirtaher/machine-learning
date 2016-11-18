## GaussianNB Deployment on Terrain Data

### ClassifyNB.py
```python
def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
        
    ### your code goes here!
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf
```

### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%231-Naive_Bayes/01-GaussianNB_Deployment_on_Terrain_Data/test.png?raw=true "Logo Title Text 1")



## Calculating NB Accuracy
### classify.py
```python
def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score

    ### create classifier
    clf = GaussianNB()

    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    # method 1
    accuracy = clf.score(features_test, labels_test)
    
    # method 2
    pred = clf.predict(features_test)
    accuracy = accuracy_score(pred, labels_test)
    
    return accuracy
```
### studentCode.py
```python
features_train, labels_train, features_test, labels_test = makeTerrainData()

def submitAccuracy():
    accuracy = NBAccuracy(features_train, labels_train, features_test, labels_test)
    return accuracy

print submitAccuracy()
```

### Output
0.884

## Project - Author ID

### nb_author_id.py

```python
#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import os
os.chdir('C:/Users/Jeff/udacity/Intro_to_Machine_Learning/ud120-projects/naive_bayes')

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

# the classifier
clf = GaussianNB()

# train
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:", round(time()-t0, 3), "s"

# predict
t0 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t0, 3), "s"

accuracy = accuracy_score(pred, labels_test)

print '\naccuracy = {0}'.format(accuracy)
```

### Output
no. of Chris training emails: 7936

no. of Sara training emails: 7884

training time: 1.614 s

training time: 0.203 s

accuracy = 0.973265073948
