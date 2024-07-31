import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
file_path = 'C:\\Users\\cardo\\Desktop\\Uni\\Magistrale\\Data_Science\\Progetto\\Progetto_Classification_Clusterin_Forecasting\\Classification-Clustering-of-Heart-Attack\\heart_new.csv'  
df = pd.read_csv(file_path)

# Distribuzione età dei campioni con istogramma
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Distribuzione età dei campioni')
plt.xlabel('Età')
plt.ylabel('Frequenza')
plt.show()

# Percentuale maschi e femmine con diagramma a torta
gender_counts = df['sex'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Percentuale maschi e femmine')
plt.show()

# cp (angina tipica) con diagramma ad albero
cp_counts = df['cp'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(cp_counts, labels=cp_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Tipi di dolore al petto (cp)')
plt.show()

# Resting blood pressure andamento ad area
plt.figure(figsize=(10, 6))
df['trtbps'].plot(kind='area', alpha=0.4, color='#ff9999')
plt.title('Andamento pressione sanguigna a riposo')
plt.xlabel('Indice')
plt.ylabel('Pressione Sanguigna a Riposo')
plt.show()

# Valori del colesterolo andamento ad area
plt.figure(figsize=(10, 6))
df['chol'].plot(kind='area', alpha=0.4, color='#66b3ff')
plt.title('Andamento valori del colesterolo')
plt.xlabel('Indice')
plt.ylabel('Colesterolo')
plt.show()

# fbs con box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='fbs', y='age', data=df)
plt.title('Box plot del livello di zucchero nel sangue a digiuno (fbs) rispetto all\'età')
plt.xlabel('Fbs (Zucchero nel Sangue a Digiuno > 120 mg/dl)')
plt.ylabel('Età')
plt.show()

# restecg con diagramma a torta
restecg_counts = df['restecg'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(restecg_counts, labels=restecg_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Risultati elettrocardiografici a riposo (restecg)')
plt.show()

# thalachh con distribuzione istogramma
plt.figure(figsize=(10, 6))
sns.histplot(df['thalach'], bins=30, kde=True, color='#ff9999')
plt.title('Distribuzione della frequenza cardiaca massima raggiunta (thalachh)')
plt.xlabel('Frequenza Cardiaca Massima')
plt.ylabel('Frequenza')
plt.show()

# exng con diagramma a torta
exng_counts = df['exang'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(exng_counts, labels=exng_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Risposta angina da esercizio (exng)')
plt.show()

# thall con diagramma ad albero
thall_counts = df['thall'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(thall_counts, labels=thall_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Risultati del test di talassemia (thall)')
plt.show()

# output con diagramma a torta
output_counts = df['output'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(output_counts, labels=output_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Output (Diagnosi di malattia)')
plt.show()

# Correlazione tra tutti i dati
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di correlazione')
plt.show()
