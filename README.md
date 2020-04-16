# Machine Learning Pipelines
 

## 1. Aim 

1. Scikit-learn Pipeline
2. Scikit-learn Feature Union
3. Pipelines and Grid Search


## 2. Using a Pipeline

Pipeline of transforms with a final estimator.

Sequentially apply a list of transforms and a final estimator. Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. The final estimator only needs to implement fit. The transformers in the pipeline can be cached using memory argument.

The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters. 

<a href='https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
'>Scikit-learn link<a>



#### 2.1 Without Pipeline
```
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    clf = RandomForestClassifier()

    # train classifier
    X_train_counts = vect.fit_transform(X_train)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    clf.fit(X_train_tfidf, y_train)

    # predict on test data
    X_test_counts = vect.transform(X_test)
    X_test_tfidf = tfidf.transform(X_test_counts)
    y_pred = clf.predict(X_test_tfidf)
```

#### 2.2 With Pipeline

```
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier()),
    ])

    # train classifier
    pipeline.fit(Xtrain)

    # evaluate all steps on test set
    predicted = pipeline.predict(Xtest)
```

## 3. Advantages of using Pipeline

- Automates repetitive steps
- Easily understandable workflow 
- Optimize workflow with **Grid Search**
- Prevents data leakage


## 4. Using Feature Union

**FEATURE UNION:** Feature union is a class in scikit-learn’s Pipeline module that concatenates results of multiple transformer objects.

This estimator applies a list of transformer objects in parallel to the input data, then concatenates the results. This is useful to combine several feature extraction mechanisms into a single transformer.

## 5. Building a custom transformer


## 6. Pipeline with grid search

Using Pipeline with GridSearchCV

As you may have seen before, grid search can be used to optimize hyper parameters of a model. Here is a simple example that uses grid search to find parameters for a support vector classifier. All you need to do is create a dictionary of parameters to search, using keys for the names of the parameters and values for the list of parameter values to check. Then, pass the model and parameter grid to the grid search object. Now when you call fit on this grid search object, it will run cross validation on all different combinations of these parameters to find the best combination of parameters for the model.

```
parameters = {
    'kernel': ['linear', 'rbf'],
    'C':[1, 10]
}

svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
```