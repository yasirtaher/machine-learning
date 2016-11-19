## SVM in SKlearn

```python
clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
```

### Output
0.92


### Kernel and Gamma
```python
clf = SVC(kernel="linear", gamma=1.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
```
### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%231-Naive_Bayes/01-GaussianNB_Deployment_on_Terrain_Data/test.png?raw=true "Logo Title Text 1")

### SVM C Parameter
```python
clf = SVC(kernel="rbf", C=10**5)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)

```

### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%231-Naive_Bayes/01-GaussianNB_Deployment_on_Terrain_Data/test.png?raw=true "Logo Title Text 1")


### SVM gamma Parameter
```python
clf = SVC(kernel="rbf", gamma=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
```

### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%231-Naive_Bayes/01-GaussianNB_Deployment_on_Terrain_Data/test.png?raw=true "Logo Title Text 1")


## Mini Project - Author ID with Naive Bayes

```python
def my_svm(kernel='linear', C=1.0):
    # the classifier
    clf = SVC(kernel=kernel, C=C)

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
    return pred

pred = my_svm()
```

### Output
no. of Chris training emails: 7936
no. of Sara training emails: 7884

### A Smaller Training Set

```python
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

pred = my_svm()
```

### Output
no. of Chris training emails: 7936
no. of Sara training emails: 7884

### Deploy an RBF Kernel

```python
pred = my_svm('rbf')
```

### Output
no. of Chris training emails: 7936
no. of Sara training emails: 7884

### Optimize C Parameter

```python
for C in [10, 100, 1000, 10000]:
    print 'C =',C,
    pred = my_svm(kernel='rbf', C=C)
    print '\n\n'
```

### Output
no. of Chris training emails: 7936
no. of Sara training emails: 7884

### Optimized RBF vs. Linear SVM: Accuracy

```python
pred = my_svm(kernel='rbf', C=10000)
```

### Output
no. of Chris training emails: 7936
no. of Sara training emails: 7884

### Extracting Predictions from an SVM

```python
for p in [10, 26, 50]:
    print pred[p]
```

### Output
1
0
1

### How many Chris emails predicted?

```python
    print sum(pred)
```

### Output
877
