import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Función para agregar etiquetas con valores encima de las barras
def agregar_etiquetas(barras):
    for barra in barras:
        altura = barra.get_height()
        plt.text(barra.get_x() + barra.get_width() / 2, altura + 1, '{:.2f}'.format(altura), ha='center', va='bottom')

df = pd.read_csv('predictions.csv')

# Datos para las barras
categorias = ["Team_Form"]

# Ancho de las barras
ancho_barras = 0.25

x = np.arange(len(categorias))

# Crear las barras
for i in x:
    categoria = df[df['method'] == categorias[i]]
    datos1 = categoria['accuracy'] * 100
    datos2 = categoria['recall'] * 100
    datos3 = categoria['f1-score'] * 100

    bar1 = plt.bar(x[i] - ancho_barras, np.mean(datos1), yerr=np.std(datos1), width=ancho_barras, color='dodgerblue')
    bar2 = plt.bar(x[i], np.mean(datos2), yerr=np.std(datos2), width=ancho_barras, color='orange')
    bar3 = plt.bar(x[i] + ancho_barras, np.mean(datos3), yerr=np.std(datos3), width=ancho_barras, color='green')

    # Configurar el gráfico
    # plt.xticks([x - ancho_barras, x, x + ancho_barras], ['Accuracy', 'Recall', 'F1-Score'])

    # Agregar etiquetas con valores encima de las barras
    agregar_etiquetas(bar1)
    agregar_etiquetas(bar2)
    agregar_etiquetas(bar3)

    plt.xticks([x[i] - ancho_barras, x[i], x[i] + ancho_barras], ['Accuracy', 'Recall', 'F1-Score'])

# Mostrar el gráfico
plt.show()
