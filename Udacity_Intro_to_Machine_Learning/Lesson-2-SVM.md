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
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%232-SVM/01-SVM_in_SKlearn/testLinear.png?raw=true)

### SVM C Parameter
```python
clf = SVC(kernel="rbf", C=10**5)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)

```

### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%232-SVM/01-SVM_in_SKlearn/testRBFC10**5.png)


### SVM gamma Parameter
```python
clf = SVC(kernel="rbf", gamma=10)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

prettyPicture(clf, features_test, labels_test)
```

### Output
![alt text](https://github.com/yasirtaher/machine-learning/blob/master/Udacity_Intro_to_Machine_Learning/Quiz/%232-SVM/01-SVM_in_SKlearn/testRBFGamma10.png)


## Mini Project - Author ID Accuracy & Timing with SVM

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

training time: 151.243 s
predicting time: 15.968 s

accuracy = 0.984072810011

### A Smaller Training Set

```python
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

pred = my_svm()
```

### Output
training time: 0.084 s
predicting time: 0.918 s

accuracy = 0.884527872582

### Deploy an RBF Kernel

```python
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

pred = my_svm('rbf')
```

### Output
training time: 0.096 s
predicting time: 1.032 s

accuracy = 0.616040955631

### Optimize C Parameter

```python
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

for C in [10, 100, 1000, 10000]:
    print 'C =',C,
    pred = my_svm(kernel='rbf', C=C)
    print '\n\n'
```

### Output
C = 10 
training time: 0.098 s
predicting time: 1.132 s

accuracy = 0.616040955631

C = 100 
training time: 0.102 s
predicting time: 1.052 s

accuracy = 0.616040955631

C = 1000 
training time: 0.089 s
predicting time: 0.983 s

accuracy = 0.821387940842

C = 10000 
training time: 0.086 s
predicting time: 0.876 s

accuracy = 0.892491467577

### Optimized RBF vs. Linear SVM: Accuracy

```python
pred = my_svm(kernel='rbf', C=10000)
```

### Output
training time: 98.278 s
predicting time: 9.973 s

accuracy = 0.990898748578

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
