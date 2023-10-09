import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

# Generar datos de ejemplo para tres conjuntos
set1 = cv2.imread('../imagenes_2/vaca.jpg').reshape(-1, 3)  # Conjunto 1
set2 = cv2.imread('../imagenes_2/pasto.jpg').reshape(-1, 3)  # Conjunto 2
set3 = cv2.imread('../imagenes_2/cielo.jpg').reshape(-1, 3)  # Conjunto 3

set1 = set1[:set1.shape[0] // 10]
set2 = set2[:set2.shape[0] // 10]
set3 = set3[:set3.shape[0] // 10]

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos de cada conjunto con colores diferentes
ax.scatter(set1[:, 0], set1[:, 1], set1[:, 2], c='r', label='Vaca')
ax.scatter(set2[:, 0], set2[:, 1], set2[:, 2], c='g', label='Pasto')
ax.scatter(set3[:, 0], set3[:, 1], set3[:, 2], c='b', label='Cielo')

ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

# Mostrar la leyenda
ax.legend()

# Mostrar el gráfico
plt.show()