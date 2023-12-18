import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_DICT = {'C': 0, 'CL': 1, 'D': 2}


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

    # Encode labels
    y_train = [CLASS_DICT[label] for label in y_train]

    return X_train, X_test, y_train


def train(X_train, y_train):
    clf = XGBClassifier()

    # Parameter search space
    param_grid = {
        'min_child_weight': [10],
        'gamma': [2],
        'subsample': [0.7],
        'colsample_bytree': [0.4],
        'max_depth': [4]
    }

    # Grid search cross-validation
    grid_search = GridSearchCV(clf, param_grid, scoring='neg_log_loss', cv=10, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the fitted model
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_clf = grid_search.best_estimator_

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


def predict(clf, X_test):
    prob_pred = clf.predict_proba(X_test)
    class_labels = ['Status_' + label for label in CLASS_DICT.keys()]

    prob_df = pd.DataFrame(prob_pred, columns=class_labels)
    prob_df['id'] = X_test['id'].reset_index(drop=True)
    columns_order = ['id'] + list(class_labels)
    prob_df = prob_df[columns_order]

    prob_df.to_csv('./data/predictions.csv', index=False)


if __name__ == '__main__':
    X_train, X_test, y_train = load_data()
    clf = train(X_train, y_train)
    #visualize_feature_importances(clf)
    predict(clf, X_test)