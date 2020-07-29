import pandas as pd

#En este caso lo primero que vemos es que no tenemos una variable respuesta a predecir, puesto que estamos en un caso no supervisado.
#Nuestro objetivo aquí por tanto es la segmentación de los clientes disponibles: hacer agrupaciones de los clientes según patrones de compra. Vemos que todas las variables disponibles pueden usarse como variables explicativas.

df = pd.read_csv("mall_customers.csv", sep=",")

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols)

# Vamos a comenzar haciendo un modelo considerando sólamente dos variables explicativas (para poder representarlo en 2D). Utilizaremos las variables "age" y "spending score".
x = df[["Age", "Spending Score (1-100)"]].values

# PROCESADO DE LOS DATOS:

"""
Para que todas las variables explicativas tengan la misma relevancia sobre el modelo y poder unificar escalas de dichas variables, realizamos una normalización de todas las variables explicativas.

Para ello utilizamos el componente MinMaxScaler del módulo 'preprocessing' de sklearn.

Esto además nos ayudará a la hora de determinar un valor para nuestro radio de vecindad o epsilon, ya que si trabajamos con dimensiones en distintas escalas es más complicado de determinar este radio que si sabemos que todas nuestras dimensiones están normalizadas entre 0 y 1.
"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

# CLUSTETING BASADO EN DENSIDADES (DBSCAN):

"""
En este caso saltamos el paso de dividir los conjuntos de datos de entrenamiento vs. validación, puesto que al ser no supervisado no podemos llevar a cabo esta fase de validación.

Calculamos el modelo DBSCAN utilizando el componente 'DBSCAN' de la librería sklearn.cluster.

Consideramos inicialmente un radio R o epsilon E de 0.1 y un número mínimo de vecinos M de 10
"""

from sklearn.cluster import DBSCAN
import numpy as np

# Calculamos el modelo
R = 0.1
M = 10

db = DBSCAN(eps=R, min_samples=M).fit(x_norm)
labels = db.labels_

# Los puntos clasificados como ruido se etiquetal con un "label" de -1
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Número de clusters obtenidos: %d' % n_clusters_)
print('Número de puntos clasificados como ruido: %d' % n_noise_)

# REPRESENTACIÓN GRÁFICA:

import matplotlib.pyplot as plt

unique_labels = set(labels)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = x_norm[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = x_norm[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Clusters: %d. Ruido: %d' % (n_clusters_, n_noise_))
plt.show()

# PROBAR DISTINTOS VALORES DE R Y DE M:

#Una vez más podemos utilizar la capacidad computacional para iterar el modelo con disintos valores de radio de vecindad R y número mínimo de vecinos M y obtener información tanto del número de clusters obtenidos como de los puntos clasificados como ruido.
#Con qué configuración quedarnos finalmente dependerá de cada set de datos y escenario.


from numpy import arange

Rs = arange(0.05, 0.20, 0.01)
Ms = range(5, 8)

conf = []
n_clusters = []
n_ruido = []

for R in Rs:
    for M in Ms:
        db = DBSCAN(eps=R, min_samples=M).fit(x_norm)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # Descartamos la configuración si hemos obtenido 0 o 1 clusters
        if n_clusters_ > 1:
            n_noise_ = list(labels).count(-1)
            conf.append((R, M))
            n_clusters.append(n_clusters_)
            n_ruido.append(n_noise_)

# Mostramos las 10 primeras configuraciones
for sconf, sclusters, sruido in zip(conf[:10], n_clusters[:10], n_ruido[:10]):
    print("Calculando DBSCAN con R=%.2f y M=%d" % (sconf[0], sconf[1]))
    print(' * Número de clusters obtenidos: %d' % sclusters)
    print(' * Número de puntos clasificados como ruido: %d' % sruido)

    # PRINT(...) ETC


# Podemos representar visualmente para cada configuración, el número de clusters y el ruido obtenido:
n_clusters_np = np.array(n_clusters, dtype="float64")
n_ruido_np = np.array(n_ruido, dtype="float64")

n_clusters_np *= (1.0/max(n_clusters_np))
n_ruido_np *= (1.0/max(n_ruido_np))

plt.figure(figsize=(8, 8))

linea_clusters, = plt.plot(range(0, len(conf)), n_clusters_np, label='N. clusters')
linea_ruido, = plt.plot(range(0, len(conf)), n_ruido_np, label='N. ruido')
plt.legend([linea_clusters, linea_ruido], ['N. clusters', 'N. ruido'])

xlabels = ["(%.2f, %d)" % (c[0], c[1]) for c in conf]
plt.xticks(range(0, len(conf)), xlabels, rotation='vertical')
plt.show()

# Podemos por ejemplo coger la configuración R = 0.08, N = 5 que minimiza la cantidad de ruido y maximiza el numero de clusters

from sklearn.cluster import DBSCAN
import numpy as np

# Calculamos el modelo
R = 0.08
M = 5

db = DBSCAN(eps=R, min_samples=M).fit(x_norm)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Número de clusters obtenidos: %d' % n_clusters_)
print('Número de puntos clasificados como ruido: %d' % n_noise_)

import matplotlib.pyplot as plt

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = x_norm[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = x_norm[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Clusters: %d. Ruido: %d' % (n_clusters_, n_noise_))
plt.show()




# CLUSTERING CON MÁS DIMENSIONES:

#El ejemplo visto ha sido considerando dos variables explicativas: "Años" y "Puntuación de ventas", pero el algoritmo funciona para cualquier dimensionalidad de entrada.

#Si consideramos ahora las variables "Años", "Puntuación de ventas" e "ingresos":

x = df[["Age", "Spending Score (1-100)", "Annual Income (k$)"]].values

# Normalizamos
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

from sklearn.cluster import DBSCAN

# Calculamos el modelo
R = 0.15
M = 10

db = DBSCAN(eps=R, min_samples=M).fit(x_norm)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Número de clusters obtenidos: %d' % n_clusters_)
print('Número de puntos clasificados como ruido: %d' % n_noise_)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = Axes3D(fig)
fig.suptitle("DBSCAN. Clusters: %d. Ruido: %d" % (n_clusters_, n_noise_), fontsize=16)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xyz = x_norm[class_member_mask & core_samples_mask]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o', edgecolors='k', color=tuple(col), s=250)

    xyz = x_norm[class_member_mask & ~core_samples_mask]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o', edgecolors='k', color=tuple(col), s=80)

    ax.set_xlabel('Años')
    ax.set_ylabel('Puntuación de gasto')
    ax.set_zlabel('Ingresos')
    ax.view_init(-10, 80)

plt.show()

## FIN



