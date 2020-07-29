import pandas as pd

"""Lectura de datos¶
Partimos de ejemplo del dataset utilizado en el módulo 2 de regresión.

El dataset refleja información de precios de viviendas y variables que pueden afectar en dicho precio, como la antigüedad de la casa, 
la distancia a la estación más cercana o el número de tiendas que tiene cerca."""

df = pd.read_csv("Técnicas supervisadas/real_estate.csv", sep=",")
print(df)

# Información estadística:

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 1) # Menos la variable respuesta

print("\nResumen estadístico de las variables:")
print("\n")
print(df.describe())


# En este caso, para realizar el análisis de componentes principales utilizaremos todas las variables explicativas disponibles (todas las del fichero excepto la variable respuesta "precio")

x_cols = list(df.columns)
x_cols.remove("Y house price of unit area")
print(x_cols)


# NORMALIZACIÓN DE LAS VARIABLES:
# Usamos el componente 'MinMaxScaler' de sklearn para normalizar nuestras variables explicativas.

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df[x_cols])
X_scaled= scaler.transform(df[x_cols])

# Podemos ver ahora que las variables se expresan en la misma escala

print(pd.DataFrame(X_scaled, columns=x_cols))


#ANÁLISIS DE COMPONENTES PRINCIPALES:

"""Aplicamos ahora el componente 'PCA' de sklearn.decomposition.
En un primer ejemplo, fijamos el número de dimensiones que queremos obtener con "n_components"""

from sklearn.decomposition import PCA

pca1=PCA(n_components=3)
pca1.fit(X_scaled)
X_pca1=pca1.transform(X_scaled)

#Podemos ver el df obtenido, y en este caso, representarlo gráficamente

df_pca_1 = pd.DataFrame(X_pca1)
print(df_pca_1)

# En un segundo ejemplo podemos decir al modelo PCA que escoja el número de variables necesarias hasta que quede explicado un porcentaje de variabilidad explicada por el conjunto inicial:

from sklearn.decomposition import PCA

pca2=PCA(0.90)
pca2.fit(X_scaled)
X_pca2=pca2.transform(X_scaled)

df_pca_2 = pd.DataFrame(X_pca2)
print(df_pca_2)


# EVALUACIÓN DEL MODELO:

#Obtenemos el parámetro "explained_variance_ratio_" del modelo para ver la cantidad de información explicada por el modelo final.
#Recordad que es un aprendizaje no supervisado: no usamos la variable respuesta "precio" para medir la bondad.

expl1 = pca1.explained_variance_ratio_
print("Modelo 1 (3 variables):", expl1)
print(" -> n.variables:", len(expl1))
print(" -> suma:", sum(expl1))

print("\n")

expl2 = pca2.explained_variance_ratio_
print("Modelo 2 (90% de varianza):", expl2)
print(" -> n.variables:", len(expl2))
print(" -> suma:", sum(expl2))


# ELBOW METHOD

#Puedo determinar en las distintas iteraciones con que número de componentes principales quedarme:

pca=PCA(n_components=6)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
ratios=pca.explained_variance_ratio_

#Autovalores obtenidos:

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('n_components')
plt.ylabel('variance')
plt.scatter(range(6), ratios)


# Ratio acumulado:
acc_ratios=[]

for i in range(6):
    if i==0:
        acc_ratios.append(ratios[i])
    else:
        acc_ratios.append(ratios[i]+acc_ratios[i-1])

plt.figure()
plt.xlabel('n_components')
plt.ylabel('variance')
plt.scatter(range(6), acc_ratios)

# REPRESENTACIÓN GRÁFICA:

# Podemos representar gráficamente nuestro modelo de 3 variables:

import matplotlib.pyplot as pl
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

# Representa los puntos de entrenamiento
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_pca_1[0],
                 df_pca_1[1],
                 df_pca_1[2], color='green')
ax.set_xlabel("0")
ax.set_ylabel("1")
ax.set_zlabel("2")
plt.show()

# access values and vectors
print(pca.components_)

print(pca.explained_variance_)














