import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patheffects as path_effects

df = pd.read_csv('predictions.csv')

estimators = df['estimators'].unique()
etas = df['eta'].unique()

grouped_by_estimators = df.groupby('estimators')

# Genera una matriz de ejemplo
time_matrix = np.zeros((5, 5))
accuracy_matrix = np.zeros((5, 5))

for estimator, grouped_estimator in grouped_by_estimators:
    grouped_by_eta = grouped_estimator.groupby('eta')
    for eta, grouped_eta in grouped_by_eta:
        estimator_idx = estimators.tolist().index(estimator)
        eta_idx = etas.tolist().index(eta)
        time_matrix[eta_idx][estimator_idx] = grouped_eta['time'].mean()
        accuracy_matrix[eta_idx][estimator_idx] = grouped_eta['accuracy'].mean()
        # time_matrix[gamma_idx][c_idx] = grouped_gamma['time'].mean()
        # accuracy_matrix[gamma_idx][c_idx] = (grouped_gamma['tp'].mean() + grouped_gamma['tn'].mean()) / (grouped_gamma['tp'].mean() + grouped_gamma['tn'].mean() + grouped_gamma['fp'].mean() + grouped_gamma['fn'].mean())

# Crea el heatmap
plt.imshow(time_matrix, cmap='YlOrRd', interpolation='nearest')

# Agrega una barra de color
plt.colorbar()

# Etiqueta los ejes x e y
plt.xlabel('Eta')
plt.ylabel('Estimators')

plt.xticks([0, 1, 2, 3, 4], etas)
plt.yticks([0, 1, 2, 3, 4], estimators)

rows, cols = time_matrix.shape

for row in range(rows):
    for col in range(cols):
        text = '{:.2f}'.format(time_matrix[row][col])
        letter_border = path_effects.withStroke(linewidth=1, foreground='black')
        plt.annotate(str(text), xy=(col, row), color='white',
                     ha='center', va='center', fontsize=12, path_effects=[letter_border])

# Muestra el gráfico
plt.savefig('time.png')
plt.clf()

############################################################################################

# Crea el heatmap
plt.imshow(accuracy_matrix, cmap='YlOrRd', interpolation='nearest')

# Agrega una barra de color
plt.colorbar()

# Etiqueta los ejes x e y
plt.xlabel('Eta')
plt.ylabel('Estimators')

plt.xticks([0, 1, 2, 3, 4], etas)
plt.yticks([0, 1, 2, 3, 4], estimators)

rows, cols = accuracy_matrix.shape

for row in range(rows):
    for col in range(cols):
        text = '{:.2f}'.format(accuracy_matrix[row][col])
        letter_border = path_effects.withStroke(linewidth=1, foreground='black')
        plt.annotate(str(text), xy=(col, row), color='white',
                     ha='center', va='center', fontsize=12, path_effects=[letter_border])

# Muestra el gráfico
plt.savefig('accuracy.png')