import pandas as pd

# Label encoding for classes
CLASS_DICT = {'C': 0, 'CL': 1, 'D': 2}

# Data files
TRAIN_DATA_FN = './data/train.csv'
TEST_DATA_FN = './data/test.csv'
ORIGINAL_DATA_FN = './data/original.csv'
PREDICTIONS_FN = './data/predictions.csv'
TRAIN_LATENT_FN = './data/train_latent.csv'
TEST_LATENT_FN = './data/test_latent.csv'


def load_data():
    """ Load kaggle data from csvs

    Returns:
        - pd.Dataframes: train features, test features, train labels
    """
    # Load raw data
    train = pd.read_csv(TRAIN_DATA_FN)
    test = pd.read_csv(TEST_DATA_FN)
    original = pd.read_csv(ORIGINAL_DATA_FN)

    # Add original data to training data
    train = pd.concat([train, original], ignore_index=True)

    # Split into features/labels
    X_train = train.drop('Status', axis=1)
    y_train = train['Status']
    X_test = test

    return X_train, X_test, y_train


def preprocess_data(X_train, X_test, y_train):
    """ Preprocess and clean data.

    Args:
        - X_train (pd.Dataframe): train features
        - X_test (pd.Dataframe): test features
        - y_train (pd.Dataframe): train labels

    Returns:
        - pd.Dataframes: cleaned data
        - list(int): IDs of test data (for final prediction output)
    """
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