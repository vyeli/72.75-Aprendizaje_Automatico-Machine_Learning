import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos desde el archivo CSV
data = pd.read_csv('../output/mixed_kernel.csv', header=None)

# Real
# sky       tp_sky_vs_cow           fn_sky_vs_cow       fn_sky_vs_grass
#           tp_sky_vs_grass              
# cow       fp_sky_vs_cow           tn_sky_vs_cow       fp_grass_vs_cow
#                                   tn_grass_vs_cow
# grass     fp_sky_vs_grass         fn_grass_vs_cow     tp_grass_vs_cow
#                                                       tn_sky_vs_grass                                                     
#
#           sky                     cow                 grass               Predicción

# Crear una matriz de confusión
confusion_matrix = np.zeros((3, 3))
runs = 10

# tp fp fn tn
for row in data.values:
    if row[0] == 'sky' and row[1] == 'cow':
        confusion_matrix[0][0] += int(row[3])
        confusion_matrix[1][0] += int(row[4])
        confusion_matrix[0][1] += int(row[5])
        confusion_matrix[1][1] += int(row[6])
    elif row[0] == 'sky' and row[1] == 'grass':
        confusion_matrix[0][0] += int(row[3])
        confusion_matrix[2][0] += int(row[4])
        confusion_matrix[0][2] += int(row[5])
        confusion_matrix[2][2] += int(row[6])
    elif row[0] == 'grass' and row[1] == 'cow':
        confusion_matrix[2][2] += int(row[3])
        confusion_matrix[1][2] += int(row[4])
        confusion_matrix[2][1] += int(row[5])
        confusion_matrix[1][1] += int(row[6])

confusion_matrix = confusion_matrix / runs
# Etiquetas de clases (opcional)
class_labels = ["Cielo", "Vaca", "Pasto"]

# Crear un gráfico de calor utilizando Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Ajustar el tamaño de fuente
sns.heatmap(confusion_matrix, annot=True, cmap="YlOrRd", fmt=".1f", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('../output/confusion_matrix.png')