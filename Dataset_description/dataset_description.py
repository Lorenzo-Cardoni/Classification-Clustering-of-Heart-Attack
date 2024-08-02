import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import sys

# Carica il dataset
file_path = 'C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\Data_Science\\Progetto\\Progetto_Classification_Clusterin_Forecasting\\Classification-Clustering-of-Heart-Attack\\heart_new.csv'  
df = pd.read_csv(file_path)


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

# Percentuale maschi e femmine con diagramma a torta
gender_counts = df['sex'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=['Maschi', 'Femmine'], autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Percentuale maschi e femmine')
#plt.show()
plt.savefig('pie_gender.png')  # Salva come immagine
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

# Livelli di zucchero nel sangue a digiuno (fbs) con diagramma a torta
fbs_counts = df['fbs'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(fbs_counts, labels=['> 120 mg/dl', '<= 120 mg/dl'], autopct='%1.1f%%', startangle=140, colors=['#FF6600', '#006699'])
plt.title('Percentuale di zucchero nel sangue a digiuno (fbs)')
#plt.show()
plt.savefig('pie_fbs.png')  # Salva come immagine
plt.close()

# restecg con diagramma a torta
restecg_counts = df['restecg'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(restecg_counts, labels=['Normale', 'Anomalia della onda ST-T', ' Ipertrofia ventricolare sinistra'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Risultati elettrocardiografici a riposo (restecg)')
#plt.show()
plt.savefig('pie_restecg.png')  # Salva come immagine
plt.close()

# Grafico: Violin plot per la frequenza cardiaca massima raggiunta (thalachh)
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['thalach'], color='#ff0000')
plt.title('Violin plot della frequenza cardiaca massima raggiunta (thalach)')
plt.xlabel('Frequenza Cardiaca Massima')
#plt.show()
plt.savefig('violin_thalachh.png')  # Salva come immagine
plt.close()

# exng con diagramma a torta
exng_counts = df['exang'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(exng_counts, labels=['NO', 'SI'], autopct='%1.1f%%', startangle=140, colors=['#6600CC', '#33CC33'])
plt.title('Angina derivante da esercizio fisico (exng)')
#plt.show()
plt.savefig('pie_exng.png')  # Salva come immagine
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
