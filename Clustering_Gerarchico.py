import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Carica i dati
data = pd.read_csv("heart_new.csv")
selected_columns = ["age", "thalach"]
data = data[selected_columns]

# Standardizza i dati
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Visualizza il dendrogramma
linked = linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrogramma Clustering Gerarchico", fontsize=16)
plt.xlabel("Punti dati", fontsize=12)
plt.ylabel("Distanza Euclidea", fontsize=12)
plt.savefig("grafici_clustering/dendrogramma.png", bbox_inches='tight')
plt.show()

# Definisci il modello di clustering gerarchico
n_clusters = 3  # lo otteniamo analizzando il dendogramma => distanza euclidea a 10 => 3 cluster
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')

# Applica il modello di clustering gerarchico ai dati
cluster_labels = clustering.fit_predict(scaled_data)

# Imposta lo stile con Seaborn
sns.set(style="darkgrid")

# Colori pastello personalizzati per ogni cluster
colors = {0: "#FF8000", 1: "#FFD700", 2: "#00CED1"}  # arancione, giallo, turchese
cluster_colors = [colors[label] for label in cluster_labels]

# Crea il grafico di clustering con colori pastello
plt.figure(figsize=(10, 7))
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=cluster_colors, s=50, alpha=0.8, edgecolor='k')

# Aggiungi titoli e etichette
plt.title("Hierarchical Clustering", fontsize=16)
plt.xlabel(f"Feature: {selected_columns[0]}", fontsize=12)
plt.ylabel(f"Feature: {selected_columns[1]}", fontsize=12)

# Crea una legenda personalizzata con gli stessi colori
legend_labels = ["Cluster A", "Cluster B", "Cluster C"]
for i, (label, color) in enumerate(colors.items()):
    plt.scatter([], [], color=color, label=legend_labels[i], s=50)
plt.legend(loc="best")

# Salva e mostra il grafico
plt.savefig("grafici_clustering/cluster_gerarchico_pastel.png", bbox_inches='tight')
plt.show()
