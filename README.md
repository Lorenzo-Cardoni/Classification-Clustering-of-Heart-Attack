# Classification-Clustering-of-Disease-Symptom-Prediction
Questa repository è relativa all'uso di alcune librerie per la creazione di modelli di classificazione e clustering.

# Requisiti
- Python 3.9 o versioni successive
- Dipendenze Python: pandas, numpy, sklearn, xgboost, seaborn, kneed, yellowbrick, pywaffle

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
usando diverse configurazioni di parametri. Per determinare il miglior valore di k per il K-means e per scegliere l'eps per il DBSCAN si è usato il metodo del gomito. Le prestazioni sono state valutate attraverso diverse metriche, tra cui la silhouette, la V-measure, l'omogeneità...
