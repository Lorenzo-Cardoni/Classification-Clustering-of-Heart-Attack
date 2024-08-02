import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Per salvare e caricare il modello

def load_and_evaluate_model(file_path, model_file):
    # Carica il dataset
    data = pd.read_csv(file_path)

    # Prepara i dati
    X = data.drop('output', axis=1)  # Features
    y = data['output']  # Target

    # Dividi i dati in set di addestramento e di test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Carica il modello salvato
    model = joblib.load(model_file)

    # Fai previsioni
    y_pred = model.predict(X_test)

    # Calcola le metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)  # Calcola la matrice di confusione

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Visualizza la matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'], 
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# Usa il percorso del tuo file CSV e il nome del file del modello salvato
file_path = 'heart_new.csv'  # Cambia con il percorso del tuo file CSV
model_file = 'best_decision_tree_model.pkl'  # Nome del file del modello salvato

load_and_evaluate_model(file_path, model_file)
