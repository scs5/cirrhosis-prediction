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
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
    }

    model = XGBClassifier(**params)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_log_loss', cv=cv, n_jobs=-1)
    return -np.mean(scores)


def train(X_train, y_train, model=None):
    if model:
        with open('best_xgb.pkl', 'rb') as model_file:
            best_clf = pickle.load(model_file)
    else:
        # Tune hyperparameters via Optuna
        study = optuna.create_study(direction="minimize")
        objective_func = lambda trial: objective(trial, X_train, y_train)
        study.optimize(objective_func, n_trials=100)

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
    feature_importances = clf.feature_importances_
    feature_names = clf.get_booster().feature_names

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Create a sideways bar plot using seaborn
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
    prob_pred = clf.predict_proba(X_test)
    class_labels = ['Status_' + label for label in CLASS_DICT.keys()]

    prob_df = pd.DataFrame(prob_pred, columns=class_labels)
    prob_df['id'] = test_IDs
    columns_order = ['id'] + list(class_labels)
    prob_df = prob_df[columns_order]

    prob_df.to_csv('./data/predictions.csv', index=False)


if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    X_train, X_test, y_train, test_IDs = preprocess_data(X_train, X_test, y_train)
    clf = train(X_train, y_train)
    #visualize_feature_importances(clf)
    predict(clf, X_test, test_IDs)