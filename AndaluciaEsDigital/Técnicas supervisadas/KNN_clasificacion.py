import pandas as pd

#la clase que queremos predecir es la especie de cada animal (representada por la columna 'class_type')
#Todas son binarias (no/si, codificadas como 0/1) menos la variable "legs" que es numérica.
df = pd.read_csv("zoo.csv", sep=",")
print(df)

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 2) # Menos el nombre del animal y la variable respuesta

print("\nTotal de especies posibles: ", df["class_type"].unique())
print("\n")
print(df.describe())

#Guardamos en una variable 'x_list' todas las columnas que representan variables explicativas.
#En nuestro caso, son todas las columnas del dataframe menos la variable respuesta 'class_type' y la variable 'animal_name', que indica el nombre del animal (único para cada observación)

x_cols = list(df.columns)
x_cols.remove("animal_name")
x_cols.remove("class_type")


# CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN:

#Usamos la función 'train_test_split' de sklearn para dividir nuestro conjunto de datos en dos subconjuntos: entrenamiento vs. validación (puesto que estamos realizando un aprendizaje supervisado).
#Utilizamos una proporción 80% de datos para entrenamiento y 20% para validación.
#El parámetro 'random_state' nos permite fijar una semilla en el procedimiento de separación de los datos para eliminar el factor aleatorio en la división (de cara a poder repetir la misma separación para la explicación del curso, pero en un caso real debería ser una separación aleatoria).

from sklearn.model_selection import train_test_split

x = df[x_cols].values
y = df["class_type"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# PROCESADO DE LOS DATOS:

#Para que todas las variables explicativas tengan la misma relevancia sobre el clasificador y poder unificar escalas de dichas variables, realizamos una normalización de todas las variables explicativas.
#Para ello utilizamos el componente MinMaxScaler del módulo 'preprocessing' de sklearn.
#Nos basamos en el conjunto de entrenamiento 'x_train' para obtener los máximos y mínimos para la normalización de valores (función 'fit_transform') y nos basamos en estos valores obtenidos para normalizar también el conjunto de validación (función 'fit')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

# K-VECINOS (KNN)

#Partimos de una suposición inicial de que queremos aplicar el procedimiento de k-vecinos con un número K de 7 (luego nos preocuparemos de intentar optimizar este valor).
#Calculamos el modelo utilizando el componente 'KNeighborsClassifier' de la librería sklearn

from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 7

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train_norm, y_train)

#Vemos que por defecto se utiliza la distancia de Minkowski para el cálculo de distancia entre puntos, pero con un parámetro p=2 es equivalente a la distancia Euclidea (con p=1, sería la distancia de Manhattan).
#Otro parámetro importante es el argumento 'weights'. Por defecto ('uniform') indica que todos los puntos tienen el mismo peso en el vecindario. Si lo cambiasemos a 'distance' daríamos mayor peso a aquellos puntos del vecindario más cercanos al punto a predecir que a los más lejanos.
#Todos los parámetros restantes pueden verse con detalle aquí: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


#Podemos obtener la precisión del modelo obtenido:
print('Precisión del clasificador sobre el conjunto de entrenamiento: {:.2f}'
     .format(knn.score(x_train_norm, y_train)))
print('Precisión del clasificador sobre el conjunto de validación: {:.2f}'
     .format(knn.score(x_test_norm, y_test)))


# PREDICCIÓN:

y_pred = knn.predict(x_test_norm)
#Podemos visualizar la equivalencia entre datos, clase real y clase predicha en forma de dataframe:
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)


# EVALUACIÓN DEL MODELO:
#Calculamos la matriz de confusión de nuestro clasificador y el parámetro F1-score como otra métrica de bondad del modelo (aparte de la precisión):
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

# CÁLCULO DE K ÓPTIMO:

#Finalmente, vamos a considerar que k=7 puede que no sea el valor óptimo para k.
#Iteramos para un valor K entre 1 y 20, calculamos el clasificador y obtenemos la precisión resultante.
#Nos quedaremos con aquel valor de k que nos proporcione mayor precisión.

scores = []
k_range = range(1, 21)

for k in k_range:
    n_neighbors = k

    # Calculamos el clasificador
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(x_train_norm, y_train)

    # Obtenemos la precisión sobre los conjuntos validación y entrenamiento
    train_acc = knn.score(x_train_norm, y_train)
    test_acc = knn.score(x_test_norm, y_test)

    # Calculamos la precisión final como la media de ambas precisiones
    k_acc = (train_acc + test_acc) / 2
    print('[K = {}] Precisión media del clasificador: {:.4f}'.format(k, k_acc))
    scores.append(k_acc)

print('\n===> Mejor K obtenido: {} (con una precisión de {:.4f})'
      .format(scores.index(max(scores)) + 1, max(scores)))

# ===> Mejor K obtenido: 2 (con una precisión de 0.9812)

#Representamos gráficamente estas precisiones obtenidas para los distintos valores de k:

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
plt.show()

# REPRESENTACIÓN VISUAL DE LAS CLASES:

#Finalmente hacemos una representación visual de las clases para el clasificador con k = 2.
#Para poder representarlo gráficamente tenemos el problema de tener que seleccionar únicamente dos variables explicativas (ejes x e y) de todas las disponibles. En nuestro caso escogemos por ejemplo la variable numérica "legs" y la binaria "hair".

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 2

# A pesar de tener multiples variables explicativas,
# para poder representarla gráficamente sólo podemos seleccionar dos
X = df[["legs", "hair"]].values
y = df["class_type"].astype('category').cat.codes.values

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue', 'red', 'green', 'purple', 'blue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue', 'red', 'green', 'purple', 'blue'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    plt.xlabel('legs')
    plt.ylabel('hair')

plt.show()

#En el resultado podemos observar que de las 7 clases resultantes (las 7 especies que hay en total) sólo podemos predecir 4 al utilizar sólo las dos variables explicativas "hair" y "legs". Necesitamos el resto de variables para poder predecir el total de especies disponibles.


## FIN




