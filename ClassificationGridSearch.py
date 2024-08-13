# Classificatori utilizzati come funzioni, ed in ognuno faccio lo standard e con grid search 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Dataset
file_path = 'heart_new.csv'
data = pd.read_csv(file_path)

# Train-Test split
X = data.drop('output', axis=1)  # Features
y = data['output']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Matrice di confusione
def save_confusion_matrix(y_test, y_pred, file_name, title):
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.title(title + ' Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig('confusion_matrix_gridsearch/' + file_name, bbox_inches='tight')
    #plt.show()


# Classificatori 
def decision_tree():
    decision_tree = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_params_dt = grid_search.best_params_
    y_pred_dt = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    dt_dict = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_))
    return y_pred_dt, dt_dict, best_params_dt

def random_forest():
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_rf = grid_search.best_params_
    y_pred_rf = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    rf_dict = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_))
    return y_pred_rf, rf_dict, best_params_rf

def svc():
    svc = SVC(random_state=42)
    param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'sigmoid']
    }
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_svc = grid_search.best_params_
    y_pred_svc = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC.png', 'Support Vector Classifier')
    return y_pred_svc, best_params_svc

def logistic_regression():
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    parameters = [{'penalty': ['l2', 'none']},
                  {'C': [1, 10, 100, 1000]},
                  {'max_iter': [100, 150, 200]},
                  {'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}]
    grid_search = GridSearchCV(estimator=logistic_regression,
                           param_grid=parameters,
                           scoring='accuracy',
                           verbose=3)
    grid_search.fit(X_train, y_train)
    best_params_lr = grid_search.best_params_
    y_pred_lr = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression')
    return y_pred_lr, best_params_lr

def xgboost():
    xgboost = XGBClassifier(random_state=42, eval_metric='logloss')
    param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_xgb = grid_search.best_params_
    y_pred_xgb = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    xg_dict = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_)) 
    return y_pred_xgb, xg_dict, best_params_xgb

def adaboost():
    adaboost = AdaBoostClassifier(random_state=42)
    param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0]
    }
    grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_ab = grid_search.best_params_
    y_pred_ab = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    ab_dict = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_))
    return y_pred_ab, ab_dict, best_params_ab

def gradient_boosting():
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 4, 5]
    }
    grid_search = GridSearchCV(estimator=gradient_boosting, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_gb = grid_search.best_params_
    y_pred_gb = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    gb_dict = dict(zip(X_train.columns, grid_search.best_estimator_.feature_importances_))
    return y_pred_gb, gb_dict, best_params_gb

def linear_discriminant():
    linear_discriminant = LinearDiscriminantAnalysis()
    param_grid = [
    {
        'solver': ['svd'],  # 'svd' does not support shrinkage
    },
    {
        'solver': ['lsqr', 'eigen'],  # 'lsqr' and 'eigen' support shrinkage
        'shrinkage': ['auto', None],
    }
]
    grid_search = GridSearchCV(estimator=linear_discriminant, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_params_ld = grid_search.best_params_
    y_pred_ld = grid_search.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ld, 'LinearDiscriminant.png', 'Linear Discriminant Analysis')
    return y_pred_ld, best_params_ld

# Esegui i modelli
y_pred_dt, dt_dict, best_params_dt = decision_tree()
y_pred_rf, rf_dict, best_params_rf = random_forest()
y_pred_svc, best_params_svc = svc()
y_pred_lr, best_params_lr = logistic_regression()
y_pred_xgb, xg_dict, best_params_xgb = xgboost()
y_pred_ab, ab_dict, best_params_ab = adaboost()
y_pred_gb, gb_dict, best_params_gb = gradient_boosting()
y_pred_ld, best_params_ld = linear_discriminant()

# heatmap feature importances
d = {'Random Forest': pd.Series(rf_dict.values(), index=rf_dict.keys()),
     'Decision Tree': pd.Series(dt_dict.values(), index=dt_dict.keys()),
     'AdaBoost': pd.Series(ab_dict.values(), index=ab_dict.keys()),
     'Gradient Boosting': pd.Series(gb_dict.values(), index=gb_dict.keys()),
     'XGBoost': pd.Series(xg_dict.values(), index=xg_dict.keys())
     }

feature_importance = pd.DataFrame(d)
sns.heatmap(feature_importance, cmap="crest")
plt.title('Feature importance by model')
plt.savefig('plots_gridsearch/Heatmap', bbox_inches='tight')
plt.show()

# Plot accuracy
def plot_accuracy_bar():
    models = ['Decision Tree', 'Random Forest', 'SVC', 'Logistic Regression', 'XGBoost', 'AdaBoost', 'Gradient Boosting', 'LinearDiscriminant']
    accuracies = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svc),
                  accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_ab), 
                  accuracy_score(y_test, y_pred_gb), accuracy_score(y_test, y_pred_ld)]

    color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']
    plt.bar(models, accuracies, color=color)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('Classifier Accuracy Comparison')
    plt.ylim([0, 1])  # Assuming accuracy values are between 0 and 1
    plt.savefig('plots_gridsearch/Accuracy', bbox_inches='tight')
    plt.show()

plot_accuracy_bar()

# Writing result in output file
output_file_path = 'results_gridsearch.txt'
with open(output_file_path, 'w') as f:
    f.write('\n<------------------------------------- Test Accuracy ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {accuracy_score(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier: {accuracy_score(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier: {accuracy_score(y_test, y_pred_svc)}\n')
    f.write(f'Logistic Regression: {accuracy_score(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier: {accuracy_score(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier: {accuracy_score(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier: {accuracy_score(y_test, y_pred_gb)}\n')
    f.write(f'LinearDiscriminant: {accuracy_score(y_test, y_pred_ld)}\n')

    f.write('\n<------------------------------------- Classification Report ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier:\n{classification_report(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier:\n{classification_report(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier:\n{classification_report(y_test, y_pred_svc)}\n')
    f.write(f'Logistic Regression:\n{classification_report(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier:\n{classification_report(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier:\n{classification_report(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier:\n{classification_report(y_test, y_pred_gb)}\n')
    f.write(f'LinearDiscriminant:\n{classification_report(y_test, y_pred_ld)}\n')

    f.write('\n<------------------------------------- Best Iperparameters ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {best_params_dt}\n')
    f.write(f'Random Forest Classifier: {best_params_rf}\n')
    f.write(f'Support Vector Machine Classifier: {best_params_svc}\n')
    f.write(f'Logistic Regression: {best_params_lr}\n')
    f.write(f'XGBoost Classifier: {best_params_xgb}\n')
    f.write(f'AdaBoost Classifier: {best_params_ab}\n')
    f.write(f'Gradient Boosting Classifier: {best_params_gb}\n')
    f.write(f'LinearDiscriminant: {best_params_ld}\n')





