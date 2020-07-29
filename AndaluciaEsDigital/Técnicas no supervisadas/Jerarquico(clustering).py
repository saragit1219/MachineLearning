import pandas as pd

#En este caso lo primero que vemos es que no tenemos una variable respuesta a predecir, puesto que estamos en un caso no supervisado.
#Nuestro objetivo aquí por tanto es la segmentación de los clientes disponibles: hacer agrupaciones de los clientes según patrones de compra. Vemos que todas las variables disponibles pueden usarse como variables explicativas.

df = pd.read_csv("mall_customers.csv", sep=",")
print(df)

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols)

# Para poder tener una prepresentación gráfica en forma de dendograma más interpretable, vamos a coger un subconjunto de los datos (en torno a 20 registros):

df = df.head(20)
rows, _ = df.shape

#Seleccionamos todas las variables explicativas que vamos a utilizar:
x = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values


# (((DEBERÍA NORMALIZAR, AQUÍ NO AFECTA PORQUE ESTÁ EN LOS MISMO RANGOS MAS O MENOS)))

# CLUSTERING JERÁRQUICO: APROXIMACIÓN AGLOMERATIVA.

#En este caso vamos a centrarnos en la más común de las aproximaciones del clustering jerárquico, que es la aglomerativa o aproximación de abajo a arriba.

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(x, 'single')
labels = df["CustomerID"].values

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=labels,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()


"""
La función dendogram puede configurarse con los siguientes parámetros:

linkage: puede ser los vistos en el módulo de "single", "complete", "average", "centroid" y alguno más
orientation: como se representa el dendograma (pero ojo, no como se calcula. Estamos en la aproximación aglomerativa en todos los casos). Puede ser "top", "bottom", "left" o "right"
distance_sort: Para cada nodo, como se representan sus descendientes: "ascending" primero (de izquierda a derecha) representa al hijo con menor distancia con sus descendentes. "descending" primero representa al hijo con mayor distancia entre sus descendentes.
color_threshold: determina un número de clusters a dibujar en diferentes colores. Por defecto si no se especifica nada se aplica la fórmula 0.7*max(Z[:,2]), es decir, 0.7 por el máximo valor de todas las observaciones.
Toda la información de estas configuraciones pueden verse en:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
"""

# EFECTOS DE LOS DISTINTOS TIPOS DE LINKAGE:


#Vamos a ver ahora, con una muestra más pequeña aún, la influencia que tienen los distintos tipos de aproximaciones en distancias entre clusters (los distintos tipos de linkages) en la construcción de clusters finales.

df = df.head(10)
x = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].values

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

labels = df["CustomerID"].values

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1, title='Single linkage')
dendrogram(linkage(x, 'single'),
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)

plt.subplot(2, 2, 2, title='Complete linkage')
dendrogram(linkage(x, 'complete'),
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)

plt.subplot(2, 2, 3, title='Average linkage')
dendrogram(linkage(x, 'average'),
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)


plt.subplot(2, 2, 4, title='Centroid linkage')
#axs[1, 0].set_title('Centroid linkage')
dendrogram(linkage(x, 'centroid'),
                orientation='top',
                labels=labels,
                distance_sort='descending',
                show_leaf_counts=True)

plt.show()

## FIN




