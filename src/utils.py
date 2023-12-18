import pandas as pd


CLASS_DICT = {'C': 0, 'CL': 1, 'D': 2}


def load_data():
    # Load raw data
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    # Split into features/labels
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    X_test = test

    return X_train, X_test, y_train


def preprocess_data(X_train, X_test, y_train):
    # Remove useless features
    useless_features = ['id']
    test_IDs = X_test['id'].to_list()
    X_train.drop(columns=useless_features, inplace=True)
    X_test.drop(columns=useless_features, inplace=True)

    # One-hot encode discrete features
    discrete_features = ['Drug', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Stage']
    X_train = pd.get_dummies(X_train, columns=discrete_features, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=discrete_features, drop_first=True)

    # Encode labels
    y_train = [CLASS_DICT[label] for label in y_train]

    return X_train, X_test, y_train, test_IDs