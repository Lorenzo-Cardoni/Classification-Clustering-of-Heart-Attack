# SVC senza Grid Search

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Passo 4: Crea e allena il modello SVC
model = SVC(kernel='linear', random_state=42)  # Puoi cambiare 'linear' con altri kernel come 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Passo 5: Predici le etichette sui dati di test
y_pred = model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')

# Visualizza il report di classificazione
print(classification_report(y_test, y_pred))

# Passo 7: Salva il modello in un file
model_filename = 'svc_model.pkl'
joblib.dump(model, model_filename)
print(f"Modello salvato come {model_filename}")

# (Opzionale) Carica il modello per verificare che funzioni
loaded_model = joblib.load(model_filename)
y_pred_loaded = loaded_model.predict(X_test)
print(f'Accuratezza del modello caricato: {accuracy_score(y_test, y_pred_loaded):.2f}')

# Passo 6: Calcola e visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# SVC con Grid Search 
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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

# Passo 4: Definisci la griglia di iperparametri
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'sigmoid']
}

# Passo 5: Configura la Grid Search
model = SVC(random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Passo 6: Allena la Grid Search
grid_search.fit(X_train, y_train)

# Passo 7: Visualizza i migliori iperparametri
print(f"Migliori iperparametri trovati: {grid_search.best_params_}")

# Passo 8: Predici le etichette sui dati di test utilizzando il miglior modello
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calcola l'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')

# Visualizza il report di classificazione
print(classification_report(y_test, y_pred))

# Passo 9: Calcola e visualizza la matrice di confusione
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Passo 10: Salva il miglior modello in un file
model_filename = 'best_svc_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Modello salvato come {model_filename}")

# (Opzionale) Carica il modello per verificare che funzioni
loaded_model = joblib.load(model_filename)
y_pred_loaded = loaded_model.predict(X_test)
print(f'Accuratezza del modello caricato: {accuracy_score(y_test, y_pred_loaded):.2f}')

# Senza Grid Search: 87% Accuracy
# Con Grid Search: 87% Accuracy