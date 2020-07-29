## EJ 4 Clustering   Sara Barrera Romero

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("credit_cards.cvs", sep=",")
print(df)

# Nos quedamos con las variables: PURCHASES y PAYMENTS:
x = df[["PURCHASES", "PAYMENTS"]].values

# K-MEANS:

#Vamos a normalizar las variables:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_norm = scaler.fit_transform(x)

from sklearn.cluster import KMeans

#Estudiamos el K-optimo:

sses=[]
ks = range(1, 11)
for k in ks:
    kmeans = KMeans(n_clusters=k).fit(x_norm)
    sses.append(kmeans.inertia_)
    print("For K={}, SSE={}".format(k, kmeans.inertia_))


#Representamos gráficamente estos valores a lo largo de la variación de K:

plt.rcParams['figure.figsize'] = (15, 5)
plt.plot(ks, sses)
plt.title('K-Means Clustering (Elbow Method)', fontsize = 20)
plt.xlabel('K')
plt.ylabel('SSEs')
plt.grid()
plt.show()


# Luego nos quedamos con K=4.

k = 4
kmeans = KMeans(n_clusters=k).fit(x_norm)

# Centroides obtenidos:

centroids = kmeans.cluster_centers_

i=0
for kcenter in centroids:
    print("Cluster n. {}. Center: {}".format(i, centroids[i]))
    i+=1

# ASIGNACIÓN DE CLUSTERS:

#Podemos ver a qué cluster pertenece cada uno de nuestros puntos obtenidos

clusters = kmeans.predict(x_norm)
df["cluster id"] = clusters
print(df)

# REPRESENTAR GRÁFICAMENTE tanto los puntos de datos como los centroides obtenidos:

x_purchases = list(map(lambda r: r[0], x_norm))
x_payments = list(map(lambda r: r[1], x_norm))
centroids = kmeans.cluster_centers_
colores = ['red', 'green', 'blue', 'purple']

clusters_colores = list(map(lambda c: colores[c], clusters))

plt.scatter(x_purchases, x_payments, c=clusters_colores, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', c=colores, s=500)
plt.xlabel('Purchases')
plt.ylabel('Payments')
plt.show()


# DBSCAN:

from sklearn.cluster import DBSCAN

# Calculamos el modelo
R = 0.08
M = 5

db = DBSCAN(eps=R, min_samples=M).fit(x_norm)
labels = db.labels_

# Los puntos clasificados como ruido se etiqueta con un "label" de -1
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Número de clusters obtenidos: %d' % n_clusters_)
print('Número de puntos clasificados como ruido: %d' % n_noise_)


# REPRESENTACIÓN GRÁFICA:


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

#Iterar por distintos valores de R y M:

from numpy import arange


Rs = arange(0.05, 0.20, 0.01)
Ms = range(2, 5)

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


# Podemos elegir R = 0.11 y M = 2 que máximiza el número de cluster y disminuye el ruido.


### JERARQUICO:

df = pd.read_csv("credit_cards.cvs", sep=",")
df = df.head(30)
rows, _ = df.shape
x = df[['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE']]


from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(x, 'complete')
labels = df["CUST_ID"].values

plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=labels,
           distance_sort='descending',
           show_leaf_counts=True)
plt.show()


# ¿Qué tipo de linkage o distancia inter-cñuster se ha utilizado? --> complete



