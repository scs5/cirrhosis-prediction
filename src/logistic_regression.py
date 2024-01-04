import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from utils import *


def train(X_train, y_train):
    """ Train logistic classification model.

    Args:
        - X_train (pd.Dataframe): train features
        - y_train (pd.Dataframe): test labels

    Returns:
        - Fitted logistic classification model
    """
    # Set up classifier pipeline
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Parameter search space
    param_grid = {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'classifier__penalty': ['l1', 'l2'],
                  'classifier__solver': ['newton-cg', 'liblinear']}

    # Grid search
    grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Train the model
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=best_params['classifier__C'], penalty=best_params['classifier__penalty']))
    ])
    clf.fit(X_train, y_train)

    # Evaluation (via cross validation)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(clf, X_train, y_train, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    print('Log Loss: %.3f +- %.3f' % (-np.mean(scores), np.std(scores)))

    return clf


def predict(clf, X_test, test_IDs):
    """ Predict test labels with classification model.

    Args:
        - clf: classification model
        - X_test (pd.Dataframe): test features
        - test_IDs (list(int)): list of test IDs
    """
    # Find probabilities of each class
    prob_pred = clf.predict_proba(X_test)
    class_labels = ['Status_' + label for label in CLASS_DICT.keys()]

    # Concatenate predictions with IDs
    prob_df = pd.DataFrame(prob_pred, columns=class_labels)
    prob_df['id'] = test_IDs
    columns_order = ['id'] + list(class_labels)
    prob_df = prob_df[columns_order]

    # Output final predictions to csv
    prob_df.to_csv(PREDICTIONS_FN, index=False)


if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train = load_data()
    X_train, X_test, y_train, test_IDs = preprocess_data(X_train, X_test, y_train)

    # Train model and make predictions
    clf = train(X_train, y_train)
    predict(clf, X_test, test_IDs)