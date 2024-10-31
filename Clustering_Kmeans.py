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
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Funzione per eseguire l'analisi della silhouette per vari numeri di cluster
def silhouette_analysis(n_clusters, X, selected_columns):
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
    colors = iter([plt.cm.tab10(i) for i in range(10)]) 
    list_colors = []
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        list_colors.append(next(colors))
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, color=list_colors[-1], edgecolor='w', alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    colors = [list_colors[cluster_labels[i]] for i in range(len(X))]
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(f"Feature space for {selected_columns[0]}")
    ax2.set_ylabel(f"Feature space for {selected_columns[1]}")

    plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}", fontsize=14, fontweight="bold")
    plt.savefig(f'grafici_clustering/silhouette_kmeans_{n_clusters}.png', bbox_inches='tight')
    plt.show()


# Funzione per applicare il metodo del gomito utilizzando diverse metriche
def elbow_method(X):    
    results = {}
    model = KMeans()
    visualizer1 = KElbowVisualizer(model, k=(3, 10), metric='distortion')
    visualizer1.fit(X)
    visualizer1.show(outpath="grafici_clustering/kelbow_distortion.png")
    results['distortion'] = visualizer1.elbow_value_
    plt.close()

    model = KMeans()
    visualizer2 = KElbowVisualizer(model, k=(3, 10), metric='calinski_harabasz')
    visualizer2.fit(X)
    visualizer2.show(outpath="grafici_clustering/kelbow_calinski.png")
    results['calinski_harabasz'] = visualizer2.elbow_value_
    plt.close()

    model = KMeans()
    visualizer3 = KElbowVisualizer(model, k=(3, 9), metric='silhouette')
    visualizer3.fit(X)
    visualizer3.show(outpath="grafici_clustering/kelbow_silhouette.png")
    results['silhouette'] = visualizer3.elbow_value_
    plt.close()

    most_common_k = Counter(results.values()).most_common(1)[0][0]
    return most_common_k, results


def bench_k_means(kmeans, name, data, labels):
    t0 = time()
    estimator = make_pipeline(None, kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]
    
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]
    results += [metrics.silhouette_score(data, estimator[-1].labels_, metric="euclidean")]

    formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*results))


data = pd.read_csv("heart_new.csv")
labels = data["output"].values
selected_columns = ["age", "thalach"]
data = data[selected_columns]

scaler = StandardScaler()
scaled_array = scaler.fit_transform(data)
data = pd.DataFrame(scaled_array, columns=data.columns)

best_k, all_k_values = elbow_method(data.values)
print("Valori di k per ciascuna metrica:", all_k_values)
print("Valore ottimale di k scelto:", best_k)

print(80 * "_")
print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI   \tAMI   \tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=best_k, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="k-means++", data=data, labels=labels)

kmeans = KMeans(init="random", n_clusters=best_k, n_init=10, random_state=0)
bench_k_means(kmeans=kmeans, name="random", data=data, labels=labels)

h = 0.02
x_min, x_max = data.values[:, 0].min() - 1, data.values[:, 0].max() + 1
y_min, y_max = data.values[:, 1].min() - 1, data.values[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
customcm = mpl.colors.ListedColormap([plt.cm.Paired(0), plt.cm.Paired(2), plt.cm.Paired(4)])

Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=customcm, aspect="auto", origin="lower")
plt.plot(data.values[:, 0], data.values[:, 1], "k.", markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="o", s=60, linewidths=3, color="black", zorder=10)
plt.title("K-means clustering")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('grafici_clustering/kmeans.png', bbox_inches='tight')
plt.show()

silhouette_analysis(best_k, data.values, selected_columns)
