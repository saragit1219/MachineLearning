import pandas as pd

#En este caso lo primero que vemos es que no tenemos una variable respuesta a predecir, puesto que estamos en un caso no supervisado.
#Nuestro objetivo aquí por tanto es la segmentación de los clientes disponibles: hacer agrupaciones de los clientes según patrones de compra. Vemos que todas las variables disponibles pueden usarse como variables explicativas.

df = pd.read_csv("mall_customers.csv", sep=",")
print(df)

# INFORMACIÓN ESTADÍSTICA:
rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols)

print(df.describe())

# Vamos a comenzar haciendo un modelo considerando sólamente dos variables explicativas (para poder representarlo en 2D). Utilizaremos las variables "age" y "spending score".

x = df[["Age", "Spending Score (1-100)"]].values

# (((DEBERÍA NORMALIZAR, AQUÍ NO AFECTA PORQUE ESTÁ EN LOS MISMO RANGOS MAS O MENOS)))


# KMEANS:

#En este caso saltamos el paso de dividir los conjuntos de datos de entrenamiento vs. validación, puesto que al ser no supervisado no podemos llevar a cabo esta fase de validación.
#Vamos a considerar un valor de K inicial que sea 5. Más adelante nos preocuparemos en ver si es el valor de K óptimo o no.
#Calculamos el modelo utilizando el componente 'KMeans' de la librería sklearn.cluster.

from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k).fit(x)

# Podemos fácilmente obtener del modelo los centroides obtenidos (serán 5 puesto que hemos especificado un k = 5)

centroids = kmeans.cluster_centers_

i=0
for kcenter in centroids:
    print("Cluster n. {}. Center: {}".format(i, centroids[i]))
    i+=1

# ASIGNACIÓN DE CLUSTERS:

#Podemos ver a qué cluster pertenece cada uno de nuestros puntos obtenidos

clusters = kmeans.predict(x)
df["cluster id"] = clusters
print(df)

# Podemos REPRESENTAR GRÁFICAMENTE tanto los puntos de datos como los centroides obtenidos:

import matplotlib.pyplot as plt

x_age = list(map(lambda r: r[0], x))
x_spending = list(map(lambda r: r[1], x))
centroids = kmeans.cluster_centers_
colores = ['red', 'green', 'blue', 'purple', 'orange']

clusters_colores = list(map(lambda c: colores[c], clusters))

plt.scatter(x_age, x_spending, c=clusters_colores, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', c=colores, s=500)
plt.xlabel('Años')
plt.ylabel('Puntuación de gasto')
plt.show()


# EVALUACIÓN DEL MODELO:

#Podemos obtener la suma de errores cuadrados de cada cluster (suma de distancias de cada punto del cluster a cada centroide) directamente del modelo:
print("Para K = 5, SSE = ", kmeans.inertia_)


# CÁLCULO DEL K ÓPTIMO:

#Finalmente, vamos a considerar que k=5 puede que no sea el valor óptimo para k.
#Iteramos para un valor K entre 2 y 20, calculamos el modelo y obtenemos el valor SSE para cada uno.


sses=[]
ks = range(1, 11)
for k in ks:
    kmeans = KMeans(n_clusters=k).fit(x)
    sses.append(kmeans.inertia_)
    print("For K={}, SSE={}".format(k, kmeans.inertia_))

#Finalmente, representamos gráficamente estos valores a lo largo de la variación de K:

plt.rcParams['figure.figsize'] = (15, 5)
plt.plot(ks, sses)
plt.title('K-Means Clustering (Elbow Method)', fontsize = 20)
plt.xlabel('K')
plt.ylabel('SSEs')
plt.grid()
plt.show()

# En este caso, podríamos quedarnos con k = 4 como el número de clusters con distancia mínima suficiente para nuestro modelo.


kmeans = KMeans(n_clusters=4).fit(x)
clusters = kmeans.predict(x)

x_age = list(map(lambda r: r[0], x))
x_spending = list(map(lambda r: r[1], x))
centroids = kmeans.cluster_centers_
colores = ['red', 'green', 'blue', 'purple']

clusters_colores = list(map(lambda c: colores[c], clusters))

plt.scatter(x_age, x_spending, c=clusters_colores, s=10)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', c=colores, s=500)
plt.xlabel('Años')
plt.ylabel('Puntuación de gasto')
plt.show()



# CLUSTERING CON MÁS DIMENSIONES:

#El ejemplo visto ha sido considerando dos variables explicativas: "Años" y "Puntuación de ventas", pero el algoritmo funciona para cualquier dimensionalidad de entrada.

#Si consideramos ahora las variables "Años", "Puntuación de ventas" e "ingresos":

# Calculamos los clusters
x = df[["Age", "Spending Score (1-100)", "Annual Income (k$)"]].values

km = KMeans(n_clusters=5).fit(x)
clusters = km.labels_
centroides = km.cluster_centers_

# Representación gráfica
from mpl_toolkits.mplot3d import Axes3D

colores = ['red', 'green', 'blue', 'cyan', 'yellow']
clusters_colores = list(map(lambda c: colores[c], clusters))

fig = plt.figure(figsize=(10, 10))
fig.suptitle("KMeans. Número de clusters obtenidos: %d" % len(km.cluster_centers_), fontsize=16)
ax = Axes3D(fig)
ax.set_xlabel('Años')
ax.set_ylabel('Puntuación de gasto')
ax.set_zlabel('Ingresos')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=clusters_colores, s=60)
ax.scatter(centroides[:, 0], centroides[:, 1], centroides[:, 2], marker='*', c=colores, s=1000)
ax.view_init(15, 170)
plt.show()

## FIN

