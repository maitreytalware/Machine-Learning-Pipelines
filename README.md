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