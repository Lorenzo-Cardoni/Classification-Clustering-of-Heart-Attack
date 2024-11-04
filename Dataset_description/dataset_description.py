import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import sys
from pywaffle import Waffle

# Carica il dataset
file_path = 'heart_new.csv'  
df = pd.read_csv(file_path)

# Waffle Chart per la distribuzione di genere (maschi e femmine)
gender_counts = df['sex'].value_counts()
fig_gender = plt.figure(
    FigureClass=Waffle, 
    rows=10,  # Maggior numero di righe per aumentare la risoluzione
    values=gender_counts, 
    labels=["{} ({})".format(k, v) for k, v in zip(['Maschi', 'Femmine'], gender_counts)],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    title={'label': 'Distribuzione di genere', 'loc': 'center', 'fontsize': 16},
    figsize=(8, 4)  # Aumenta le dimensioni del grafico
)
fig_gender.savefig('Dataset_description/waffle_gender.png', bbox_inches='tight')
plt.close(fig_gender)

# Waffle Chart per i livelli di zucchero nel sangue a digiuno (fbs)
fbs_counts = df['fbs'].value_counts()
fig_fbs = plt.figure(
    FigureClass=Waffle, 
    rows=10,
    values=fbs_counts, 
    labels=["{} ({})".format(k, v) for k, v in zip(['> 120 mg/dl', '<= 120 mg/dl'], fbs_counts)],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    title={'label': 'Percentuale di zucchero nel sangue a digiuno (fbs)', 'loc': 'center', 'fontsize': 16},
    figsize=(8, 4)
)
fig_fbs.savefig('Dataset_description/waffle_fbs.png', bbox_inches='tight')
plt.close(fig_fbs)

# Waffle Chart per i risultati elettrocardiografici a riposo (restecg)
restecg_counts = df['restecg'].value_counts()
fig_restecg = plt.figure(
    FigureClass=Waffle, 
    rows=10,
    values=restecg_counts, 
    labels=["{} ({})".format(k, v) for k, v in zip(['Normale', 'Anomalia della onda ST-T', 'Ipertrofia ventricolare sinistra'], restecg_counts)],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    title={'label': 'Risultati elettrocardiografici a riposo (restecg)', 'loc': 'center', 'fontsize': 16},
    figsize=(12, 8)
)
fig_restecg.savefig('Dataset_description/waffle_restecg.png', bbox_inches='tight')
plt.close(fig_restecg)

# Waffle Chart per angina derivante da esercizio fisico (exang)
exng_counts = df['exang'].value_counts()
fig_exng = plt.figure(
    FigureClass=Waffle, 
    rows=10,
    values=exng_counts, 
    labels=["{} ({})".format(k, v) for k, v in zip(['No', 'Sì'], exng_counts)],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    title={'label': 'Angina derivante da esercizio fisico (exng)', 'loc': 'center', 'fontsize': 16},
    figsize=(8, 4)
)
fig_exng.savefig('Dataset_description/waffle_exng.png', bbox_inches='tight')
plt.close(fig_exng)

# Mappare i valori di `output` a etichette descrittive e creare il Waffle Chart per `output`
output_labels = {
    0: '< 50% possibilità di malattie cardiache',
    1: '> 50% possibilità di malattie cardiache'
}
df['output_label'] = df['output'].map(output_labels)
output_counts = df['output_label'].value_counts()
fig_output = plt.figure(
    FigureClass=Waffle, 
    rows=10,
    values=output_counts, 
    labels=["{} ({})".format(k, v) for k, v in zip(output_counts.index, output_counts)],
    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    title={'label': 'Distribuzione delle diagnosi di possibili malattie cardiache (output)', 'loc': 'center', 'fontsize': 16},
    figsize=(12, 8)
)
fig_output.savefig('Dataset_description/waffle_output.png', bbox_inches='tight')
plt.close(fig_output)

print("I grafici a waffle sono stati salvati come immagini nella cartella 'Dataset_description'.")

# Carica il dataset (sostituisci il percorso con quello del tuo file)
# Assumendo che il dataset sia in formato CSV
data = df

# Crea il grafico scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='thalach', hue='output', palette='Set1', data=data, alpha=0.7)

# Imposta il titolo e le etichette
plt.title('Age vs. Maximum Heart Rate')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate (thalach)')

# Mostra la legenda
plt.legend(title='Output')

# Salva il grafico se necessario
plt.savefig('grafico_age_vs_thalach.png', bbox_inches='tight')

# Mostra il grafico
plt.show()



# Reindirizza l'output su un file di testo
with open('dataset_info.txt', 'w') as f:
    # Reindirizza sys.stdout su file
    old_stdout = sys.stdout
    sys.stdout = f

    # Stampa informazioni del dataset
    df_info = df.info()
    df_description = df.describe(include='all')  # Include tutte le colonne, anche non numeriche
    print("Informazioni del Dataset:\n")
    print(df_info)
    print("\nDescrizione Statistica del Dataset:\n")
    print(df_description.to_string())  # Usa to_string() per visualizzare tutto il contenuto

    # Ripristina sys.stdout
    sys.stdout = old_stdout

print("Le informazioni sono state salvate in 'dataset_info.txt'")


# Distribuzione età dei campioni con istogramma
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
mean_age = df['age'].mean()
plt.axvline(mean_age, color='r', linestyle='--', label=f'Media: {mean_age:.2f}')
plt.title('Distribuzione età dei campioni')
plt.xlabel('Età')
plt.ylabel('Frequenza')
#plt.show()
plt.savefig('histogram_age.png')  # Salva come immagine
plt.close()

# Mappare i valori di `cp` a etichette descrittive
cp_labels = {
    0: 'Angina tipica',
    1: 'Angina atipica',
    2: 'Dolore Non Anginoso',
    3: 'Asintomatico'
}
df['cp_label'] = df['cp'].map(cp_labels)


# Treemap per i tipi di dolore al petto (cp)
cp_counts = df['cp_label'].value_counts().reset_index()
cp_counts.columns = ['cp_label', 'count']
fig_cp = px.treemap(cp_counts, path=['cp_label'], values='count', color='cp_label', title='Distribuzione dei tipi di dolore al petto (cp)')
#fig_cp.show()
fig_cp.update_traces(textinfo='label+percent entry+value')
fig_cp.write_image('treemap_cp.png')  # Salva come immagine


# Pressione sanguigna a riposo con diagramma a violino
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['trtbps'], color='#66b3ff')
plt.title('Violin plot della pressione sanguigna a riposo')
plt.xlabel('Pressione Sanguigna a Riposo')
#plt.show()
plt.savefig('violin_trestbps.png')  # Salva come immagine
plt.close()


# Valori del colesterolo con diagramma a violino
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['chol'], color='#ffff00')
plt.title('Violin plot dei valori del colesterolo')
plt.xlabel('Colesterolo')
#plt.show()
plt.savefig('violin_chol.png')  # Salva come immagine
plt.close()

# Grafico: Violin plot per la frequenza cardiaca massima raggiunta (thalachh)
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['thalach'], color='#ff0000')
plt.title('Violin plot della frequenza cardiaca massima raggiunta (thalach)')
plt.xlabel('Frequenza Cardiaca Massima')
#plt.show()
plt.savefig('violin_thalachh.png')  # Salva come immagine
plt.close()

# Mappare i valori di `thall` a etichette descrittive
thall_labels = {
    0: 'Nullo',
    1: 'Difettoso Fisso',
    2: 'Normale',
    3: 'Difettoso Reversibile'
}
df['thall_label'] = df['thall'].map(thall_labels)


# thall con diagramma ad albero
thall_counts = df['thall_label'].value_counts().reset_index()
thall_counts.columns = ['thall_label', 'count']
#fig.show()
fig_thall = px.treemap(thall_counts, path=['thall_label'], values='count', title='Distribuzione dei risultati del test di talassemia (thall)')
fig_thall.update_traces(textinfo='label+percent entry+value')
fig_thall.write_image('treemap_thall.png')  # Salva come immagine

# Mappare i valori di `output` a etichette descrittive
output_labels = {
    0: '< 50% possibilità di malattie cardiache',
    1: '> 50% possibilità di malattie cardiache'
}
df['output_label'] = df['output'].map(output_labels)

# output con diagramma a barre
plt.figure(figsize=(10, 6))
sns.countplot(x=df['output_label'], palette='pastel')
plt.title('Distribuzione delle diagnosi di possibili malattie cardiache(output)')
plt.xlabel('')
plt.ylabel('Conteggio')
#plt.show()
plt.savefig('bar_output.png')  # Salva come immagine
plt.close()

# Seleziona solo le colonne numeriche
numeric_df = df.select_dtypes(include='number')

# Grafico: Correlazione tra tutti i dati
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='RdBu', center=0, linewidths=0.5, fmt='.2f')
plt.title('Matrice di correlazione')
plt.savefig('heatmap_correlation.png')  # Salva come immagine
plt.close()

print("Tutti i grafici sono stati salvati come immagini.")
