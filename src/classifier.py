import numpy as numpy
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_data():
    # Split into features/labels
    train = pd.read_csv('./data/train.csv')
    X_test = pd.read_csv('./data/test.csv')
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    
    # One-hot encode discrete features
    discrete_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
    X_train = pd.get_dummies(X_train, columns=discrete_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=discrete_features, drop_first=True)

    return X_train, X_test, y_train


def train(X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def predict(clf, X_test):
    prob_pred = clf.predict_proba(X_test)
    class_labels = ['Status_' + label for label in clf.classes_]

    prob_df = pd.DataFrame(prob_pred, columns=class_labels)
    prob_df['id'] = X_test['id'].reset_index(drop=True)
    columns_order = ['id'] + list(class_labels)
    prob_df = prob_df[columns_order]

    prob_df.to_csv('./data/predictions.csv', index=False)


if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    clf = train(X_train, y_train)
    predict(clf, X_test)