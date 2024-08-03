import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Per salvare e caricare il modello

# Passo 1: Carica il dataset
file_path = 'heart_new.csv'
data = pd.read_csv(file_path)

# Passo 2: Prepara i dati
X = data.drop('output', axis=1)  # Features
y = data['output']  # Target

# Passo 3: Dividi i dati in set di addestramento e di test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crea il modello di regressione logistica
model = LogisticRegression(max_iter=1000)  # max_iter è il numero massimo di iterazioni per l'algoritmo di ottimizzazione

# Allena il modello
model.fit(X_train, y_train)

# Predici le etichette sui dati di test
y_pred = model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')

# Visualizza il report di classificazione
print(classification_report(y_test, y_pred))

# Calcola la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

# Passo 8: Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Salva il modello in un file
model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)