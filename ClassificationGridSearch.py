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
    plt.savefig('confusion_matrix/' + file_name, bbox_inches='tight')
    plt.show()


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
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    dt_dict = dict(zip(X_train.columns, decision_tree.feature_importances_))
    return y_pred_dt, dt_dict

def random_forest():
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    rf_dict = dict(zip(X_train.columns, random_forest.feature_importances_))
    return y_pred_rf, rf_dict

def svc():
    svc = SVC(random_state=42)
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC.png', 'Support Vector Classifier')
    return y_pred_svc

def logistic_regression():
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression')
    return y_pred_lr

def xgboost():
    xgboost = XGBClassifier(random_state=42)
    xgboost.fit(X_train, y_train)
    y_pred_xgb = xgboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    xg_dict = dict(zip(X_train.columns, xgboost.feature_importances_))
    return y_pred_xgb, xg_dict

def adaboost():
    adaboost = AdaBoostClassifier(random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred_ab = adaboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    ab_dict = dict(zip(X_train.columns, adaboost.feature_importances_))
    return y_pred_ab, ab_dict

def gradient_boosting():
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    gradient_boosting.fit(X_train, y_train)
    y_pred_gb = gradient_boosting.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    gb_dict = dict(zip(X_train.columns, gradient_boosting.feature_importances_))
    return y_pred_gb, gb_dict

# Esegui i modelli
y_pred_dt, dt_dict = decision_tree()
y_pred_rf, rf_dict = random_forest()
y_pred_svc = svc()
y_pred_lr = logistic_regression()
y_pred_xgb, xg_dict = xgboost()
y_pred_ab, ab_dict = adaboost()
y_pred_gb, gb_dict = gradient_boosting()

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
plt.savefig('plots/Heatmap', bbox_inches='tight')
plt.show()

# comparisons
fig, ax = plt.subplots(figsize=(11, 3))
ax = sns.lineplot(x=y_test, y=y_pred_xgb, label='XGBoost')
ax1 = sns.lineplot(x=y_test, y=y_pred_gb, label='GradientBoosting')
ax2 = sns.lineplot(x=y_test, y=y_pred_ab, label='AdaBoost')

ax.set_xlabel('y_test', color='g')
ax.set_ylabel('y_pred', color='g')
plt.title('Comparison between models')
plt.savefig('plots/Comparison', bbox_inches='tight')
plt.show()

# Plot accuracy
def plot_accuracy_bar():
    models = ['Decision Tree', 'Random Forest', 'Support Vector Machine', 'Logistic Regression', 'XGBoost', 'AdaBoost', 'Gradient Boosting']
    accuracies = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf), accuracy_score(y_test, y_pred_svc),
                  accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_xgb), accuracy_score(y_test, y_pred_ab), accuracy_score(y_test, y_pred_gb)]

    color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f']
    plt.bar(models, accuracies, color=color)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=90)
    plt.title('Classifier Accuracy Comparison')
    plt.ylim([0, 1])  # Assuming accuracy values are between 0 and 1
    plt.savefig('plots/Accuracy', bbox_inches='tight')
    plt.show()

plot_accuracy_bar()

# Writing result in output file
output_file_path = 'results.txt'
with open(output_file_path, 'w') as f:
    f.write('\n<------------------------------------- Test Accuracy ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {accuracy_score(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier: {accuracy_score(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier: {accuracy_score(y_test, y_pred_svc)}\n')
    f.write(f'Logistic Regression: {accuracy_score(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier: {accuracy_score(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier: {accuracy_score(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier: {accuracy_score(y_test, y_pred_gb)}\n')

    f.write('\n<------------------------------------- Classification Report ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier:\n{classification_report(y_test, y_pred_dt)}\n')
    f.write(f'Random Forest Classifier:\n{classification_report(y_test, y_pred_rf)}\n')
    f.write(f'Support Vector Machine Classifier:\n{classification_report(y_test, y_pred_svc)}\n')
    f.write(f'Logistic Regression:\n{classification_report(y_test, y_pred_lr)}\n')
    f.write(f'XGBoost Classifier:\n{classification_report(y_test, y_pred_xgb)}\n')
    f.write(f'AdaBoost Classifier:\n{classification_report(y_test, y_pred_ab)}\n')
    f.write(f'Gradient Boosting Classifier:\n{classification_report(y_test, y_pred_gb)}\n')