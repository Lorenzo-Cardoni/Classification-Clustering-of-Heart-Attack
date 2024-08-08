from PIL import Image
import matplotlib.pyplot as plt

# Percorso della directory con le matrici di confusione salvate
image_files = [
    'confusion_matrix/AdaBoost.png',
    'confusion_matrix/DecisionTree.png',
    'confusion_matrix/GradientBoosting.png',
    'confusion_matrix/LinearDiscriminant.png',
    'confusion_matrix/LogisticRegression.png',
    'confusion_matrix/RandomForest.png',
    'confusion_matrix/SVC.png',
    'confusion_matrix/XGBoost.png',
]

# Carica le immagini e ottieni le loro dimensioni
images = [Image.open(img_file) for img_file in image_files]
widths, heights = zip(*(i.size for i in images))

# Calcola la dimensione totale della figura
total_width = max(widths) * 4  # 4 colonne
max_height = max(heights) * 2  # 2 righe

# Crea una nuova immagine bianca con le dimensioni calcolate
new_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

# Incolla le immagini nella nuova immagine
x_offset = 0
y_offset = 0
for i, img in enumerate(images):
    new_image.paste(img, (x_offset, y_offset))
    x_offset += img.width
    if (i + 1) % 4 == 0:  # Se siamo alla fine di una riga
        x_offset = 0
        y_offset += img.height

# Salva l'immagine combinata
new_image.save('confusion_matrix\Combined_Confusion_Matrices.png')

# Mostra l'immagine combinata
plt.imshow(new_image)
plt.axis('off')  # Disabilita gli assi
plt.show()