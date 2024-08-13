import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def elbow_method(dataset, n_knn, config):
    neighbors = NearestNeighbors(n_neighbors=n_knn)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    kl = KneeLocator(range(1, len(distances) + 1), distances, curve="convex")
    kl.plot_knee(figsize=(9, 6))
    plt.savefig(f"grafici/elbow_dbscan_{config}.png", bbox_inches='tight')
    plt.show()

    return kl.elbow, kl.knee_y

def select_parameter(eps, dataset, config):
    eps_to_test = [round(e, 3) for e in np.arange(max(eps - 1, 0.1), eps + 1, 0.1)]
    min_samples_to_test = range(5, 50, 5)

    results_noise = pd.DataFrame(
        data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
        columns=min_samples_to_test,
        index=eps_to_test
    )

    results_clusters = pd.DataFrame(
        data=np.zeros((len(eps_to_test), len(min_samples_to_test))),  # Empty dataframe
        columns=min_samples_to_test,
        index=eps_to_test
    )

    iter_ = 0

    print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
    print("-" * 65)

    for e in eps_to_test:
        for min_samples in min_samples_to_test:
            iter_ += 1
            # Calcolo le metriche
            noise_metric, cluster_metric = get_metrics(e, min_samples, dataset, iter_)

            # Inserisco i risultati nei relativi dataframe
            results_noise.loc[e, min_samples] = noise_metric
            results_clusters.loc[e, min_samples] = cluster_metric

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    sns.heatmap(results_noise, annot=True, ax=ax1, cbar=False).set_title("METRIC: Mean Noise Points Distance")
    sns.heatmap(results_clusters, annot=True, ax=ax2, cbar=False).set_title("METRIC: Number of clusters")

    ax1.set_xlabel("min_samples")
    ax2.set_xlabel("min_samples")
    ax1.set_ylabel("EPSILON")
    ax2.set_ylabel("EPSILON")

    plt.tight_layout()
    plt.savefig(f"grafici/parametri_dbscan_{config}.png", bbox_inches='tight')
    plt.show()

def get_metrics(eps, min_samples, dataset, iter_):
    # Fitting ======================================================================
    dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model_.fit(dataset)

    # Mean Noise Point Distance metric =============================================
    noise_indices = dbscan_model_.labels_ == -1

    if True in noise_indices:
        neighbors = NearestNeighbors(n_neighbors=6).fit(dataset)
        distances, indices = neighbors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))

    print("%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s" % (
        iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters))

    return noise_mean_distance, number_of_clusters

file = 'heart.csv'
dataset = pd.read_csv(file)
dataset = dataset.iloc[:, 1:]

# Aggiorna la colonna target se necessario
labels = dataset["output"].values
dataset = dataset.drop("output", axis=1)

# Normalizzazione
scaler = StandardScaler()
scaled_array = scaler.fit_transform(dataset)
data_scaled = pd.DataFrame(scaled_array, columns=dataset.columns)

# Utilizza il metodo dell'elbow per determinare un buon valore di eps
n = 5
x, eps = elbow_method(data_scaled, n, config='data_scaled')
print("eps=" + str(eps))

# Testa con valori diversi di eps e min_samples
select_parameter(eps, data_scaled, config='data_scaled')

# Utilizza i migliori parametri trovati per l'algoritmo DBSCAN
min_points = 15
db = DBSCAN(eps=eps, min_samples=min_points).fit(data_scaled)

ymeans = db.labels_

# Calcola metriche
n_clusters_ = len(set(ymeans)) - (1 if -1 in ymeans else 0)
n_noise_ = list(ymeans).count(-1)

homogeneity = metrics.homogeneity_score(ymeans, labels)
completeness = metrics.completeness_score(ymeans, labels)
v_measure = metrics.v_measure_score(ymeans, labels)
ari = metrics.adjusted_rand_score(ymeans, labels)
ami = metrics.adjusted_mutual_info_score(ymeans, labels)
silhouette = metrics.silhouette_score(data_scaled, ymeans)
m_calinski = metrics.calinski_harabasz_score(data_scaled, ymeans)
m_bouldin = metrics.davies_bouldin_score(data_scaled, ymeans)

unique, counts = np.unique(ymeans, return_counts=True)

df = pd.DataFrame({'Scaling': ['Standard'],
                   'knn': n,
                   'eps': eps,
                   'min points': min_points,
                   'n_cluster': n_clusters_,
                   'homogeneity': homogeneity,
                   'completeness': completeness,
                   'v_measure': v_measure,
                   'ari': ari,
                   'ami': ami,
                   'calinksi': m_calinski,
                   'bouldin': m_bouldin,
                   'silhouette': silhouette,
                   'sample': str(dict(zip(unique, counts)))
                   })

df.to_csv('metrics_DBSCAN.csv', header=None, index=False, mode='a')

print(f"Homogeneity: {homogeneity:.3f}")
print(f"Completeness: {completeness:.3f}")
print(f"V-measure: {v_measure:.3f}")
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Adjusted Mutual Information: {ami:.3f}")
print(f"Calinski Harabasz Score: {m_calinski:.3f}")
print(f"Davies Bouldin Score: {m_bouldin:.3f}")
print(f"Silhouette Coefficient: {silhouette:.3f}")
print(f"Estimated number of clusters: {n_clusters_}")
print(f"Estimated number of noise points: {n_noise_}")
