import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from yellowbrick.cluster import KElbowVisualizer


# 1. Caricamento dei dati
HAD = pd.read_csv('heart.csv')

# 2. Visualizzazione della distribuzione delle caratteristiche
sns.set_theme()
lst = ["Age", "Sex", "Chest pain", "Resting blood pressure", "Cholestrol", "Fasting blood suger >120?", 
       "resting electrocardiographic result", "Max heart rate", "Excercise induced angina", "old Peak",
       "slp", "caa", "thall", "Whether high risk?"]
lst1 = list(HAD.columns)
plt.figure(figsize=(17,22))
for i in range(len(lst)):
    plt.subplot(5, 3, i+1)
    plt.hist(HAD[lst1[i]], bins=15, color="green")
    plt.xlabel(lst[i])
    plt.ylabel("Counts")
    plt.savefig("grafici_cluster/Istrogrammi.png")
plt.show()

# 3. Visualizzazione della matrice di correlazione
plt.figure(figsize=(15,15))
sns.heatmap(HAD.corr(), annot=True, cmap="Blues_r")
plt.savefig("grafici_cluster/heatmap.png")
plt.show()

# 4. Esplorazione e rimozione delle colonne con valori binari
binary_cols = ["sex", "fbs", "exng", "output"]
for col in binary_cols:
    print(HAD[col].value_counts())

# 5. Rimozione delle colonne non utili per il clustering
HADE = HAD.drop(binary_cols, axis=1)

# 6. Standardizzazione del dataset
scaler = StandardScaler(with_mean=False, with_std=True)
HADES = scaler.fit_transform(HADE)

# 7. Applicazione del K-means con n_clusters=2
KMC = KMeans(n_clusters=2)
KMC.fit(HADES)
center_age = [KMC.cluster_centers_[0][0], KMC.cluster_centers_[1][0]]
center_thalachh = [KMC.cluster_centers_[0][5], KMC.cluster_centers_[1][5]]

# 8. Visualizzazione della previsione (n_clusters=2) rispetto ai dati reali
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
sns.scatterplot(x="age", y="thalachh", data=HADE, hue=KMC.labels_, s=40)
plt.title("Predicted outcome based on clustering")
plt.xlabel("Age")
plt.ylabel("Max heart Rate")

plt.subplot(1, 2, 2)
sns.scatterplot(x="age", y="thalachh", data=HADE, hue=HAD["output"], s=40)
plt.title("Actual outcome")
plt.xlabel("Age")
plt.ylabel("Max heart Rate")
plt.savefig("grafici_cluster/Previsione Cluster.png")
plt.show()

# 9. Calcolo dell'accuratezza
def accr(actual_labels, predicted_labels):
    count = sum(1 for a, p in zip(actual_labels, predicted_labels) if a == p)
    accuracy = count / len(actual_labels)
    return 1 - accuracy

print(f"Accuracy (1 - error): {accr(list(HAD['output']), list(KMC.labels_))}")

# 10. Metodo del gomito per determinare il numero ottimale di cluster
metric = []
k_values = list(range(2, 10))
for k in k_values:
    KMC = KMeans(n_clusters=k)
    KMC.fit(HADES)
    metric.append(KMC.inertia_)

plt.figure(figsize=(10,10))
plt.plot(k_values, metric, 'bx-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances to nearest centroid')
plt.title('Elbow Method for Optimal k')
plt.savefig("grafici_cluster/Elbow Method Cluster.png")
plt.show()

# 11. Analisi silhouette per diverse quantità di cluster
def silhouette_analysis(range_n_clusters, X):
    for n_clusters in range_n_clusters:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        cluster_labels = clusterer.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
        
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        colors = iter([plt.cm.Paired(i) for i in range(0, 20)])
        color_list = []
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color_list.append(next(colors))
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, color=color_list[-1], edgecolor='w', alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = [color_list[cluster_labels[c]] for c in range(len(X))]
        ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=np.array(colors), edgecolor="k")
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}", fontsize=14, fontweight="bold")
        plt.savefig(f'grafici_cluster/Grafico_{n_clusters}.png')
        plt.show()

# 12. Riduzione della dimensionalità con PCA e clustering K-means
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(HADE)

# 13. Applicazione dell'analisi silhouette
silhouette_analysis(range(2, 6), data_reduced)
