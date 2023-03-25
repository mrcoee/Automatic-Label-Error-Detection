from sklearn import linear_model

def classification_fit_and_predict(X_train, y_train, ls, X_test):
    model = linear_model.LogisticRegressionCV(Cs=ls, penalty='l1', solver='saga', random_state=0, max_iter=1000, tol=1e-3)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    y_train_pred = model.predict_proba(X_train)
    return y_test_pred, y_train_pred