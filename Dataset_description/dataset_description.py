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
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Percentuale maschi e femmine')
#plt.show()
plt.savefig('pie_gender.png')  # Salva come immagine
plt.close()

# Treemap per i tipi di dolore al petto (cp)
cp_counts = df['cp'].value_counts().reset_index()
cp_counts.columns = ['cp', 'count']
fig_cp = px.treemap(cp_counts, path=['cp'], values='count', title='Distribuzione dei tipi di dolore al petto (cp)')
#fig_cp.show()
fig_cp = px.treemap(cp_counts, path=['cp'], values='count', title='Distribuzione dei tipi di dolore al petto (cp)')
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
sns.violinplot(y=df['chol'], color='#ff9999')
plt.title('Violin plot dei valori del colesterolo')
plt.xlabel('Colesterolo')
#plt.show()
plt.savefig('violin_chol.png')  # Salva come immagine
plt.close()

# Livelli di zucchero nel sangue a digiuno (fbs) con diagramma a torta
fbs_counts = df['fbs'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(fbs_counts, labels=['> 120 mg/dl', '<= 120 mg/dl'], autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Percentuale di zucchero nel sangue a digiuno (fbs)')
#plt.show()
plt.savefig('pie_fbs.png')  # Salva come immagine
plt.close()

# restecg con diagramma a torta
restecg_counts = df['restecg'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(restecg_counts, labels=restecg_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Risultati elettrocardiografici a riposo (restecg)')
#plt.show()
plt.savefig('pie_restecg.png')  # Salva come immagine
plt.close()

# Grafico: Violin plot per la frequenza cardiaca massima raggiunta (thalachh)
plt.figure(figsize=(10, 6))
sns.violinplot(y=df['thalach'], color='#ff9999')
plt.title('Violin plot della frequenza cardiaca massima raggiunta (thalach)')
plt.xlabel('Frequenza Cardiaca Massima')
#plt.show()
plt.savefig('violin_thalachh.png')  # Salva come immagine
plt.close()

# exng con diagramma a torta
exng_counts = df['exang'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(exng_counts, labels=exng_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Risposta angina da esercizio (exng)')
#plt.show()
plt.savefig('pie_exng.png')  # Salva come immagine
plt.close()

# thall con diagramma ad albero
thall_counts = df['thall'].value_counts().reset_index()
thall_counts.columns = ['thall', 'count']
fig = px.treemap(thall_counts, path=['thall'], values='count', title='Distribuzione dei risultati del test di talassemia (thall)')
#fig.show()
fig_thall = px.treemap(thall_counts, path=['thall'], values='count', title='Distribuzione dei risultati del test di talassemia (thall)')
fig_thall.write_image('treemap_thall.png')  # Salva come immagine

# output con diagramma a barre
plt.figure(figsize=(10, 6))
sns.countplot(x=df['output'], palette='pastel')
plt.title('Distribuzione delle diagnosi di malattia (output)')
plt.xlabel('Diagnosi di Malattia')
plt.ylabel('Conteggio')
#plt.show()
plt.savefig('bar_output.png')  # Salva come immagine
plt.close()

# Correlazione tra tutti i dati
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di correlazione')
#plt.show()
plt.savefig('heatmap_correlation.png')  # Salva come immagine
plt.close()

print("Tutti i grafici sono stati salvati come immagini.")
