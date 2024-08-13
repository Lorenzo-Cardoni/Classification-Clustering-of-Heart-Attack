import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Definizione delle funzioni per l'analisi silhouette, metodo del gomito e benchmarking K-means

# Funzione per eseguire l'analisi della silhouette per vari numeri di cluster
def silhouette_analysis(range_n_clusters, X):
    for n_clusters in range_n_clusters:
        # Configurazione delle subplot per visualizzare i grafici dell'analisi silhouette
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(12, 6)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
        # Creazione del modello KMeans con il numero di cluster corrente
        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)
        cluster_labels = clusterer.fit_predict(X)
        
        # Calcolo del punteggio silhouette medio per il numero di cluster corrente
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(f"For n_clusters = {n_clusters}, The average silhouette_score is : {silhouette_avg}")
        
        # Calcolo dei valori silhouette per ciascun campione
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        y_lower = 10
        colors = iter([plt.cm.Paired(i) for i in range(0, 20)])  # Genera una lista di colori per i cluster
        list = []
        
        # Creazione del grafico silhouette
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            list.append(next(colors))
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, color=list[-1], edgecolor='w', alpha=0.7)
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        # Configurazione e visualizzazione del grafico silhouette
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # Creazione del grafico di visualizzazione dei dati clusterizzati
        colors = []
        for c in range(0, len(X)):
            colors.append(list[cluster_labels[c]])

        ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=np.array(colors), edgecolor="k")
        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        # Salvataggio e visualizzazione del grafico complessivo
        plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}", fontsize=14, fontweight="bold")
        plt.savefig(f'grafici/silhouette_kmeans_{n_clusters}.png', bbox_inches='tight')
        plt.show()


# Funzione per applicare il metodo del gomito utilizzando diverse metriche
def elbow_method(X):    
    # Visualizzazione con la metrica 'distortion'
    model = KMeans()
    visualizer1 = KElbowVisualizer(model, k=(3, 10), metric='distortion')
    visualizer1.fit(X)
    visualizer1.show(outpath="grafici/kelbow_distortion.png")
    plt.close()

    # Visualizzazione con la metrica 'calinski_harabasz'
    model = KMeans()
    visualizer2 = KElbowVisualizer(model, k=(3, 10), metric='calinski_harabasz')
    visualizer2.fit(X)
    visualizer2.show(outpath="grafici/kelbow_calinksi.png")
    plt.close()

    # Visualizzazione con la metrica 'silhouette'
    model = KMeans()
    visualizer3 = KElbowVisualizer(model, k=(3, 9), metric='silhouette')
    visualizer3.fit(X)
    visualizer3.show(outpath="grafici/kelbow_silhouette.png")
    plt.close()

# Funzione per eseguire il benchmarking del K-means con diverse inizializzazioni
def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(None, kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    
    # Definizione delle metriche di clustering
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [metrics.silhouette_score(data, estimator[-1].labels_, metric="euclidean")]

    # Formattazione dei risultati e stampa
    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))


# Lettura dei dati dal file CSV
data = pd.read_csv("heart.csv")

# Separazione delle etichette dal dataset
labels = data["output"].values
data = data.drop("output", axis=1)

# Standardizzazione dei dati per il K-means
scaler = StandardScaler()
scaled_array = scaler.fit_transform(data)
data = pd.DataFrame(scaled_array, columns=data.columns)

# Applicazione del metodo del gomito per determinare il numero ottimale di cluster
elbow_method(data.values)

print(80 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI   \tAMI   \tsilhouette")

# Benchmarking K-means con inizializzazione 'k-means++'
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

# Benchmarking K-means con inizializzazione 'random'
kmeans = KMeans(init="random", n_clusters=3, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

# Riduzione della dimensionalità con PCA
pca = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=3, n_init='auto', random_state=0)
bench_k_means(kmeans=kmeans, name="PCA2-based", data=pca, labels=labels)

# Visualizzazione dei risultati della riduzione di dimensionalità e clustering
pca = PCA(n_components=2)
pca_x = pca.fit_transform(data)
data_reduced = pd.DataFrame(pca_x).values
kmeans = KMeans(init="k-means++", n_clusters=3, n_init='auto', random_state=0)
labels = kmeans.fit(data_reduced)

# Impostazione dei limiti e delle dimensioni della griglia per la visualizzazione
h = 0.02
x_min, x_max = data_reduced[:, 0].min() - 1, data_reduced[:, 0].max() + 1
y_min, y_max = data_reduced[:, 1].min() - 1, data_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predizione delle classi su tutta la griglia per creare una visualizzazione
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
customcm = mpl.colors.ListedColormap([plt.cm.Paired(0), plt.cm.Paired(2), plt.cm.Paired(4)])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=customcm, aspect="auto", origin="lower")
plt.plot(data_reduced[:, 0], data_reduced[:, 1], "k.", markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=60, linewidths=3, color="black", zorder=10)
plt.title("K-means clustering")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('grafici/kmeans.png', bbox_inches='tight')
plt.show()

# Esecuzione dell'analisi silhouette per visualizzare come si distribuiscono i cluster
silhouette_analysis(range(2,5), data_reduced)
