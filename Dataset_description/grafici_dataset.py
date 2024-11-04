import pandas as pd
import matplotlib.pyplot as plt
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
