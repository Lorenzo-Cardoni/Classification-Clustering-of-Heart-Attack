# DECISION TREE CLASSIFIER SENZA GRID SEARCH
'''
DECISION TREE CLASSIFIER SENZA GRID SEARCH
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Passo 1: Carica il dataset
file_path = 'heart_new.csv'
data = pd.read_csv(file_path)

# Passo 2: Prepara i dati
# Supponiamo che la colonna 'output' sia la variabile target e il resto siano features
X = data.drop('output', axis=1)  # Features
y = data['output']  # Target

# Passo 3: Dividi i dati in set di addestramento e di test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Crea e allena il modello
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Passo 5: Fai previsioni e valuta il modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Passo 6: Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

'''

# DECISION TREE CLASSIFIER CON GRID SEARCH
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Per salvare e caricare il modello

def main():
    # Passo 1: Carica il dataset
    file_path = 'heart_new.csv' 
    data = pd.read_csv(file_path)

    # Passo 2: Prepara i dati
    X = data.drop('output', axis=1)  # Features
    y = data['output']  # Target

    # Passo 3: Dividi i dati in set di addestramento e di test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Passo 4: Definisci il modello e i parametri per Grid Search
    model = DecisionTreeClassifier()

    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Passo 5: Configura Grid Search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Passo 6: Addestra il modello con Grid Search
    grid_search.fit(X_train, y_train)

    # Passo 7: Ottieni il miglior modello e valuta le sue prestazioni
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Calcola le metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)  # Calcola la matrice di confusione

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Passo 8: Visualizza la matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    joblib.dump(best_model, 'best_decision_tree_model.pkl')
    print("Model saved to 'best_decision_tree_model.pkl'")

if __name__ == "__main__":
    main()

# Senza Grid Search: 82% Accuracy
# Con Grid Search: 88.5% Accuracy