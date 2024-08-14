# Classificatori utilizzati come funzioni, ed in ognuno faccio lo standard e con grid search 
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve
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
from sklearn.metrics import roc_curve, roc_auc_score

# Dataset
file_path = 'heart_new.csv'
data = pd.read_csv(file_path)

# Train-Test split
X = data.drop('output', axis=1)  # Features
y = data['output']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Learning Cruve
def save_learning_curve(model):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), random_state=42)

    # Calcolare la media e la deviazione standard per le curve di apprendimento
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot della curva di apprendimento
    plt.figure()
    model_name = model.__class__.__name__  # Ottiene solo il nome della classe del modello
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(f'plots/LearningCurve_{model_name}.png')
    plt.close()

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
    decision_tree.fit(X_train, y_train)
    y_pred_dt = decision_tree.predict(X_test)
    save_confusion_matrix(y_test, y_pred_dt, 'DecisionTree.png', 'Decision Tree Classifier')
    save_learning_curve(decision_tree)
    dt_dict = dict(zip(X_train.columns, decision_tree.feature_importances_))
    return y_pred_dt, dt_dict

def random_forest():
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train, y_train)
    y_pred_rf = random_forest.predict(X_test)
    save_confusion_matrix(y_test, y_pred_rf, 'RandomForest.png', 'Random Forest Classifier')
    save_learning_curve(random_forest)
    rf_dict = dict(zip(X_train.columns, random_forest.feature_importances_))
    return y_pred_rf, rf_dict

def svc():
    svc = SVC(kernel='linear', random_state=42)
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    save_confusion_matrix(y_test, y_pred_svc, 'SVC.png', 'Support Vector Classifier')
    save_learning_curve(svc)
    return y_pred_svc

def logistic_regression():
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)
    logistic_regression.fit(X_train, y_train)
    y_pred_lr = logistic_regression.predict(X_test)
    save_confusion_matrix(y_test, y_pred_lr, 'LogisticRegression.png', 'Logistic Regression')
    save_learning_curve(logistic_regression)
    # Ottieni i parametri predefiniti
    params = logistic_regression.get_params()
    print(params)
    return y_pred_lr

def xgboost():
    xgboost = XGBClassifier(random_state=42, eval_metric='logloss')
    xgboost.fit(X_train, y_train)
    y_pred_xgb = xgboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_xgb, 'XGBoost.png', 'XGBoost Classifier')
    save_learning_curve(xgboost)
    xg_dict = dict(zip(X_train.columns, xgboost.feature_importances_)) 
    return y_pred_xgb, xg_dict

def adaboost():
    adaboost = AdaBoostClassifier(random_state=42)
    adaboost.fit(X_train, y_train)
    y_pred_ab = adaboost.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ab, 'AdaBoost.png', 'AdaBoost Classifier')
    save_learning_curve(adaboost)
    ab_dict = dict(zip(X_train.columns, adaboost.feature_importances_))
    return y_pred_ab, ab_dict

def gradient_boosting():
    gradient_boosting = GradientBoostingClassifier(random_state=42)
    gradient_boosting.fit(X_train, y_train)
    y_pred_gb = gradient_boosting.predict(X_test)
    save_confusion_matrix(y_test, y_pred_gb, 'GradientBoosting.png', 'Gradient Boosting Classifier')
    save_learning_curve(gradient_boosting)
    gb_dict = dict(zip(X_train.columns, gradient_boosting.feature_importances_))
    return y_pred_gb, gb_dict

def linear_discriminant():
    linear_discriminant = LinearDiscriminantAnalysis()
    linear_discriminant.fit(X_train, y_train)
    y_pred_ld = linear_discriminant.predict(X_test)
    save_confusion_matrix(y_test, y_pred_ld, 'LinearDiscriminant.png', 'Linear Discriminant Analysis')
    save_learning_curve(linear_discriminant)
    return y_pred_ld

# Esegui i modelli
y_pred_dt, dt_dict = decision_tree()
y_pred_rf, rf_dict = random_forest()
y_pred_svc = svc()
y_pred_lr = logistic_regression()
y_pred_xgb, xg_dict = xgboost()
y_pred_ab, ab_dict = adaboost()
y_pred_gb, gb_dict = gradient_boosting()
y_pred_ld = linear_discriminant()

# heatmap feature importances
d = {'RandomForest': pd.Series(rf_dict.values(), index=rf_dict.keys()),
     'DecisionTree': pd.Series(dt_dict.values(), index=dt_dict.keys()),
     'AdaBoost': pd.Series(ab_dict.values(), index=ab_dict.keys()),
     'GradientBoosting': pd.Series(gb_dict.values(), index=gb_dict.keys()),
     'XGBoost': pd.Series(xg_dict.values(), index=xg_dict.keys())
     }

feature_importance = pd.DataFrame(d)
sns.heatmap(feature_importance, cmap="crest")
plt.title('Feature importance by model')
plt.savefig('plots/Heatmap', bbox_inches='tight')
plt.show()

# Plot accuracy
def plot_accuracy_bar():
    models = ['DecisionTree', 'RandomForest', 'SVC', 'LogisticRegression', 'XGBoost', 'AdaBoost', 'GradientBoosting', 'LinearDiscriminant']
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


# Funzione per tracciare la curva ROC
def plot_roc_curve(fpr, tpr, roc_auc, model_name, ax):
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Funzione per calcolare e tracciare tutte le curve ROC in un'unica immagine
def plot_all_roc_curves():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calcola e traccia la ROC per Decision Tree
    dt_model = DecisionTreeClassifier().fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'Decision Tree', ax)

    # Calcola e traccia la ROC per Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'Random Forest', ax)

    # Calcola e traccia la ROC per SVC
    svc_model = SVC(kernel='linear', random_state=42, probability=True).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, svc_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, svc_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'SVC', ax)

    # Calcola e traccia la ROC per Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, lr_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'Logistic Regression', ax)

    # Calcola e traccia la ROC per XGBoost
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'XGBoost', ax)

    # Calcola e traccia la ROC per AdaBoost
    ab_model = AdaBoostClassifier(random_state=42).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, ab_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, ab_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'AdaBoost', ax)

    # Calcola e traccia la ROC per Gradient Boosting
    gb_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, gb_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, gb_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'Gradient Boosting', ax)

    # Calcola e traccia la ROC per Linear Discriminant
    lda_model = LinearDiscriminantAnalysis().fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, lda_model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, lda_model.predict_proba(X_test)[:, 1])
    plot_roc_curve(fpr, tpr, roc_auc, 'Linear Discriminant', ax)

    # Configura il plot
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc='lower right')

    plt.savefig('plots/ROC_Curves.png', bbox_inches='tight')
    plt.show()

# Esegui il plot delle curve ROC
plot_all_roc_curves()
