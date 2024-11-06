# Classification-Clustering-of-Disease-Symptom-Prediction
Questa repository è relativa all'uso di alcune librerie per la creazione di modelli di classificazione e clustering.

# Requisiti
Python 3.9 o versioni successive
Dipendenze Python: pandas, numpy, sklearn, xgboost, seaborn, kneed, yellowbrick, pywaffle



#Clustering
I modelli addestrati per il clustering sono:

DBSCAN,
K-means
usando diverse configurazioni di parametri. Per determinare il miglior valore di k per il K-means e per scegliere l'eps per il DBSCAN si è usato il metodo del gomito. Le prestazioni sono state valutate attraverso diverse metriche, tra cui la silhouette, la V-measure, l'omogeneità...
