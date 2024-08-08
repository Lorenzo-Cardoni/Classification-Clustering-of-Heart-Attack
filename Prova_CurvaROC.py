# Classificatori utilizzati come funzioni, ed in ognuno faccio lo standard e con grid search 
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.interpolate import interp1d

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
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred_dt = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    return model

def random_forest():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_rf = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    return model

def svc():
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred_svc = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC.png', 'Support Vector Classifier')
    return model

def logistic_regression():
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred_lr = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression')
    return model

def xgboost():
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred_xgb = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    return model

def adaboost():
    model = AdaBoostClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_ab = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    return model

def gradient_boosting():
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred_gb = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    return model

def linear_discriminant():
    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred_ld = model.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ld, 'LinearDiscriminant.png', 'Linear Discriminant Analysis')
    return model

# Esegui i modelli
decision_tree_model = decision_tree()
random_forest_model = random_forest()
svc_model = svc()
logistic_regression_model = logistic_regression()
xgboost_model = xgboost()
adaboost_model = adaboost()
gradient_boosting_model = gradient_boosting()
linear_discriminant_model = linear_discriminant()

def smooth_roc_curve(fpr, tpr, num_points=1000):
    # Filtra i punti duplicati
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]

    # Interpolazione
    x_new = np.linspace(unique_fpr.min(), unique_fpr.max(), num_points)
    f_interp = interp1d(unique_fpr, unique_tpr, kind='linear', fill_value='extrapolate')
    tpr_smooth = f_interp(x_new)
    return x_new, tpr_smooth

def plot_smooth_roc_curve(y_test, y_pred_prob, classifier_name, color):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    fpr_smooth, tpr_smooth = smooth_roc_curve(fpr, tpr)
    
    plt.plot(fpr_smooth, tpr_smooth, color=color, label=f'{classifier_name} (AUC = {auc:.2f})')

def plot_all_roc_curves(X_test, y_test):
    # Ottieni le probabilità dei modelli
    y_pred_prob_dt = decision_tree_model.predict_proba(X_test)[:, 1]
    y_pred_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]
    y_pred_prob_svc = svc_model.decision_function(X_test)  # Per SVC
    y_pred_prob_lr = logistic_regression_model.predict_proba(X_test)[:, 1]
    y_pred_prob_xgb = xgboost_model.predict_proba(X_test)[:, 1]
    y_pred_prob_ab = adaboost_model.predict_proba(X_test)[:, 1]
    y_pred_prob_gb = gradient_boosting_model.predict_proba(X_test)[:, 1]
    y_pred_prob_ld = linear_discriminant_model.predict_proba(X_test)[:, 1]

    # Imposta la dimensione del grafico
    plt.figure(figsize=(10, 8))

    # Traccia le curve ROC con smoothing
    plot_smooth_roc_curve(y_test, y_pred_prob_dt, 'Decision Tree', 'blue')
    plot_smooth_roc_curve(y_test, y_pred_prob_rf, 'Random Forest', 'green')
    plot_smooth_roc_curve(y_test, y_pred_prob_svc, 'SVC', 'red')
    plot_smooth_roc_curve(y_test, y_pred_prob_lr, 'Logistic Regression', 'orange')
    plot_smooth_roc_curve(y_test, y_pred_prob_xgb, 'XGBoost', 'purple')
    plot_smooth_roc_curve(y_test, y_pred_prob_ab, 'AdaBoost', 'brown')
    plot_smooth_roc_curve(y_test, y_pred_prob_gb, 'Gradient Boosting', 'pink')
    plot_smooth_roc_curve(y_test, y_pred_prob_ld, 'Linear Discriminant', 'cyan')

    # Imposta le etichette e mostra il grafico
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC per vari classificatori')
    plt.legend(loc='lower right')
    plt.savefig('Curva ROC.png')
    plt.show()


# Esegui tutti i modelli e visualizza la Curva ROC
plot_all_roc_curves(X_test, y_test)

# Plot accuracy
def plot_accuracy_bar():
    models = ['Decision Tree', 'Random Forest', 'SVC', 'Logistic Regression', 'XGBoost', 'AdaBoost', 'Gradient Boosting', 'LinearDiscriminant']
    accuracies = [accuracy_score(y_test, decision_tree_model.predict(X_test)),
                  accuracy_score(y_test, random_forest_model.predict(X_test)),
                  accuracy_score(y_test, svc_model.predict(X_test)),
                  accuracy_score(y_test, logistic_regression_model.predict(X_test)),
                  accuracy_score(y_test, xgboost_model.predict(X_test)),
                  accuracy_score(y_test, adaboost_model.predict(X_test)),
                  accuracy_score(y_test, gradient_boosting_model.predict(X_test)),
                  accuracy_score(y_test, linear_discriminant_model.predict(X_test))]

    color = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00']
    plt.bar(models, accuracies, color=color)
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')
    plt.ylim([0, 1])  # Assuming accuracy values are between 0 and 1
    plt.savefig('plots/Accuracy', bbox_inches='tight')
    plt.show()

plot_accuracy_bar()

# Writing result in output file
output_file_path = 'results.txt'
with open(output_file_path, 'w') as f:
    f.write('\n<------------------------------------- Test Accuracy ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier: {accuracy_score(y_test, decision_tree_model.predict(X_test))}\n')
    f.write(f'Random Forest Classifier: {accuracy_score(y_test, random_forest_model.predict(X_test))}\n')
    f.write(f'Support Vector Machine Classifier: {accuracy_score(y_test, svc_model.predict(X_test))}\n')
    f.write(f'Logistic Regression: {accuracy_score(y_test, logistic_regression_model.predict(X_test))}\n')
    f.write(f'XGBoost Classifier: {accuracy_score(y_test, xgboost_model.predict(X_test))}\n')
    f.write(f'AdaBoost Classifier: {accuracy_score(y_test, adaboost_model.predict(X_test))}\n')
    f.write(f'Gradient Boosting Classifier: {accuracy_score(y_test, gradient_boosting_model.predict(X_test))}\n')
    f.write(f'LinearDiscriminant: {accuracy_score(y_test, linear_discriminant_model.predict(X_test))}\n')

    f.write('\n<------------------------------------- Classification Report ------------------------------------->\n\n')
    f.write(f'Decision Tree Classifier:\n{classification_report(y_test, decision_tree_model.predict(X_test))}\n')
    f.write(f'Random Forest Classifier:\n{classification_report(y_test, random_forest_model.predict(X_test))}\n')
    f.write(f'Support Vector Machine Classifier:\n{classification_report(y_test, svc_model.predict(X_test))}\n')
    f.write(f'Logistic Regression:\n{classification_report(y_test, logistic_regression_model.predict(X_test))}\n')
    f.write(f'XGBoost Classifier:\n{classification_report(y_test, xgboost_model.predict(X_test))}\n')
    f.write(f'AdaBoost Classifier:\n{classification_report(y_test, adaboost_model.predict(X_test))}\n')
    f.write(f'Gradient Boosting Classifier:\n{classification_report(y_test, gradient_boosting_model.predict(X_test))}\n')
    f.write(f'LinearDiscriminant:\n{classification_report(y_test, linear_discriminant_model.predict(X_test))}\n')
