import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke

def annotate_heatmap(ax, data=None, fmt="d", cmap="Blues", threshold=None, **kwargs):
    """
    Annotate a heatmap.
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Data must be a 2D array.")
    if threshold is not None:
        data = np.copy(data)
        data[data < threshold] = None

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = data[i][j]
            color = "white"
            text = ax.text(j + 0.5, i + 0.5, f"{value}", ha="center", va="center", color=color, fontweight="bold", fontsize=16, **kwargs)
            text.set_path_effects([withStroke(linewidth=1.7, foreground='black')])

# Calcular la matriz de confusión
conf_matrix = [[111.4, 8, 19.6], [49.4, 8.8, 19.8], [41.8, 7.6, 37.6]]

# Etiquetas para las clases
class_names = ['Local', 'Empate', 'Visitante']

# Crear un mapa de calor con Seaborn
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.15)
heatmap = sns.heatmap(conf_matrix, fmt='d', cmap='YlOrRd', xticklabels=class_names, yticklabels=class_names)

annotate_heatmap(heatmap, data=conf_matrix)

linewidth = 0.5

# Agregar líneas separadoras
plt.axhline(y=0, color='black',linewidth=linewidth)
plt.axhline(y=1, color='black',linewidth=linewidth)
plt.axhline(y=2, color='black',linewidth=linewidth)
plt.axhline(y=3, color='black',linewidth=linewidth)
plt.axvline(x=0, color='black',linewidth=linewidth)
plt.axvline(x=1, color='black',linewidth=linewidth)
plt.axvline(x=2, color='black',linewidth=linewidth)
plt.axvline(x=3, color='black',linewidth=linewidth)

plt.show()