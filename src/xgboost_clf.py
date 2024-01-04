import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import optuna
import pickle


def objective(trial, X, y):
    """
    Args:
        - trial (optuna.trial.Trial): Optuna training trial
        - X (pd.Dataframe): features
        - y (pd.Dataframe): labels

    Returns:
        - float: negative log loss
    """
    # Tunable parameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    }

    # 10-fold CV score (negative log loss)
    model = XGBClassifier(**params)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    return -np.mean(scores)


def train(X_train, y_train, model=None, n_trials=50):
    """ Finds best hyperparameters for an XGBoost model 
        and retrains it on the entire training data.

    Args:
        - X_train (pd.Dataframe): training features
        - y_train (pd.Dataframe): training labels
        - model (string): name of model file to load, 
            defaults to tuning/training a new model
        - n_trials (int): number of trials to tune via Optuna
    """
    # Load existing model if specified
    if model:
        with open('./models/' + model, 'rb') as model_file:
            best_clf = pickle.load(model_file)
    # Else, tune and train a new model
    else:
        # Tune hyperparameters via Optuna
        study = optuna.create_study(direction="minimize")
        objective_func = lambda trial: objective(trial, X_train, y_train)
        study.optimize(objective_func, n_trials=n_trials)

        # Retrieve best parameters
        best_params = study.best_params
        print("Best Parameters:", best_params)

        # Train classifier with best parameters
        best_clf = XGBClassifier(**best_params)
        best_clf.fit(X_train, y_train)

        # Save the trained model using pickle
        with open('./models/xgb.pkl', 'wb') as model_file:
            pickle.dump(best_clf, model_file)

    # Evaluation (via cross-validation)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(best_clf, X_train, y_train, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    print('Log Loss: %.3f +- %.3f' % (-np.mean(scores), np.std(scores)))

    return best_clf


def visualize_feature_importances(clf):
    """ Plot feature importances of XGBoost.

    Args:
        -clf: classifier with feature importances
            (e.g. XGBoostClassifier, RandomForestClassifier)
    """
    # Retreive feature names and importances from model
    feature_importances = clf.feature_importances_
    feature_names = clf.get_booster().feature_names
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.set_theme()
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df, orient='h', hue='Feature', palette='viridis')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importances')
    plt.tight_layout()
    plt.savefig('./figures/feature_importances.png', bbox_inches='tight')
    plt.show()


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
    clf = train(X_train, y_train, n_trials=50)
    predict(clf, X_test, test_IDs)

    # Visualize feature importances
    visualize_feature_importances(clf)