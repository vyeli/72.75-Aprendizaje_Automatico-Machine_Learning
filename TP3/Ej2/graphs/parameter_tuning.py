import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects

df = pd.read_csv('../output/cow_vs_grass_C_poly.csv')

degree_values = df['degree'].unique()
C_values = df['C'].unique()

grouped_by_C = df.groupby('C')

# Genera una matriz de ejemplo
time_matrix = np.zeros((3, 5))
accuracy_matrix = np.zeros((3, 5))

for C, grouped_C in grouped_by_C:
    grouped_by_gamma = grouped_C.groupby('degree')
    for gamma, grouped_gamma in grouped_by_gamma:
        c_idx = C_values.tolist().index(C)
        gamma_idx = degree_values.tolist().index(gamma)
        time_matrix[gamma_idx][c_idx] = grouped_gamma['time'].mean()
        accuracy_matrix[gamma_idx][c_idx] = (grouped_gamma['tp'].mean() + grouped_gamma['tn'].mean()) / (grouped_gamma['tp'].mean() + grouped_gamma['tn'].mean() + grouped_gamma['fp'].mean() + grouped_gamma['fn'].mean())

# Crea el heatmap
plt.imshow(time_matrix, cmap='viridis', interpolation='nearest')

# Agrega una barra de color
plt.colorbar()

# Etiqueta los ejes x e y
plt.ylabel('Degree')
plt.xlabel('C')

plt.xticks([0, 1, 2, 3, 4], C_values)
plt.yticks([0, 1, 2], degree_values)

rows, cols = time_matrix.shape

for row in range(rows):
    for col in range(cols):
        text = '{:.2f}'.format(time_matrix[row][col])
        letter_border = path_effects.withStroke(linewidth=1, foreground='black')
        plt.annotate(str(text), xy=(col, row), color='white',
                     ha='center', va='center', fontsize=12, path_effects=[letter_border])

# Muestra el gráfico
plt.savefig('../output/cow_vs_grass_poly_time.png')
plt.clf()

############################################################################################

# Crea el heatmap
plt.imshow(accuracy_matrix, cmap='viridis', interpolation='nearest')

# Agrega una barra de color
plt.colorbar()

# Etiqueta los ejes x e y
plt.ylabel('Degree')
plt.xlabel('C')

plt.xticks([0, 1, 2, 3, 4], C_values)
plt.yticks([0, 1, 2], degree_values)

rows, cols = accuracy_matrix.shape

for row in range(rows):
    for col in range(cols):
        text = '{:.2f}'.format(accuracy_matrix[row][col])
        letter_border = path_effects.withStroke(linewidth=1, foreground='black')
        plt.annotate(str(text), xy=(col, row), color='white',
                     ha='center', va='center', fontsize=12, path_effects=[letter_border])

# Muestra el gráfico
plt.savefig('../output/cow_vs_grass_poly_accuracy.png')