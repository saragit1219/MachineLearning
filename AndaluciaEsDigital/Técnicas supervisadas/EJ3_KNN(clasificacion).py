# EJ 3 KNN # Sara Barrera Romero

# NOTA: Vamos a usar en la clasificación de los datos de entrenamiento y validación random_state=0 para que en los 4 pasos de el ejercicio 3 tengamos
# la misma clasificación de los datos de entrenamiento y validación.

import pandas as pd

# Queremos predecir si una seta es venenosa o no a partir de una serie de datos:

df = pd.read_csv("mushrooms.csv", sep=",")
print(df)

#Guardamos en una variable 'x_list' todas las columnas que representan variables explicativas. Que en nuestro caso son todas menos class.

x_cols = list(df.columns)
x_cols.remove("class")
print(x_cols)

# Conjuntos de entrenamiento y validación:

# Repartimos nuestro conjunto de datos en conjunto de entrenamiento y conjunto de validación.
# (Hacemos una partición como en la teoría de 80% - 20%)
from sklearn.model_selection import train_test_split

x = df[x_cols].values
y = df["class"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Ahora vamos a realizar una normalización de los datos para que todas las variables explicativas tengan la misma relevancia. Para ello usamos MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

# Estudiamos cual es el k óptimo para realizar el método KNN:
from sklearn.neighbors import KNeighborsClassifier
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

#Representamos gráficamente estas precisiones obtenidas para los distintos valores de k:

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
plt.show()

# Aunque salen muchos valores de K entre 1 y 20 donde la precisión es de 1.0, en nuestro caso vamos a quedarnos con K=1

# MODELO K-VECINOS (KNN):

n_neighbors = 1

knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train_norm, y_train)

#Precisión del modelo obtenido:
print('Precisión del clasificador sobre el conjunto de entrenamiento: {:.2f}'
     .format(knn.score(x_train_norm, y_train)))
print('Precisión del clasificador sobre el conjunto de validación: {:.2f}'
     .format(knn.score(x_test_norm, y_test)))

# Se puede hacer una PREDICCIÓN con los datos de validación:

y_pred = knn.predict(x_test_norm)
# Visualizar la equivalencia entre datos, clase real y clase predicha en forma de dataframe:
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)



# EVALUACIÓN DEL MODELO:
# Matriz de confusión de nuestro clasificador y parámetro F1-score:
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))


# Por último podemos hacer una representación visual de las clases:

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 1

# solo podemos represntar dos variables explicativas:
X = df[["odor", "population"]].values
y = df["class"].astype('category').cat.codes.values

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
    plt.xlabel('odor')
    plt.ylabel('population')

plt.show()