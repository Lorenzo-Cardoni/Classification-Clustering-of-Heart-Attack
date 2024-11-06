# Classification-Clustering-of-Heart-Attack-Prediction
Questa repository è relativa all'uso di alcune librerie per la creazione di modelli di classificazione e clustering.

# Requisiti
- Python 3.9 o versioni successive
- Dipendenze Python: pandas, numpy, sklearn, xgboost, seaborn, kneed, yellowbrick, pywaffle

# Il dataset
Questo dataset è utilizzato per analizzare i fattori di rischio associati agli attacchi cardiaci e per costruire modelli predittivi di supporto clinico. Contiene variabili demografiche e cliniche relative ai pazienti, come:

- age: Età del paziente
- sex: Sesso (1 = maschio, 0 = femmina)
- cp: Tipo di dolore al petto (0–3, dove 3 indica un dolore più severo)
- trtbps: Pressione sanguigna a riposo (in mm Hg)
- chol: Colesterolo sierico (in mg/dl)
- fbs: Glicemia a digiuno (> 120 mg/dl, 1 = vero, 0 = falso)
- restecg: Risultati dell'elettrocardiogramma a riposo (0–2)
- thalach: Frequenza cardiaca massima raggiunta
- exang: Angina indotta da esercizio (1 = sì, 0 = no)
- oldpeak: Depressione del segmento ST indotta dall'esercizio rispetto a riposo
- slp: Pendenza del tratto ST
- caa: Numero di vasi principali colorati dalla fluoroscopia (0–3)
- thall: Tipo di talassemia (1–3)
- output: Variabile di target (1 = rischio di attacco cardiaco, 0 = nessun rischio)

Dopo una prima fase di ETL e di analisi descrittiva, si sono addestrati i modelli.

# Classificazione
Gli algoritmi di machine learning che sono stati utilizzati al fine di ottenere dei classificatori sono:

- Decision Tree,
- Random Forest,
- Gradient Boosting,
- XGBoost,
- AdaBoost,
- Support Vector Classification,
- Logistic Regression.
- Linear Discriminant
Per trovare la migliore combinazione di iperparametri si è utilizzata la Grid Search.

# Clustering
I modelli addestrati per il clustering sono:

- DBSCAN,
- K-means
- Clustering Gerarchico

usando diverse configurazioni di parametri. Per determinare il miglior valore di k per il K-means e per scegliere l'eps per il DBSCAN si è usato il metodo del gomito. Le prestazioni sono state valutate attraverso diverse metriche, tra cui la silhouette, la V-measure, l'omogeneità...
