import pandas as pd

#Un vistazo a los datos nos permite ver que la clase que queremos predecir es la especie de cada animal (representada por la columna 'class_type').
#Para ello podemos valernos de todas las variables explicativas disponibles.
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


# MÁQUINA DE VECTOR SOPORTE (SVM):

from sklearn.svm import SVC

svc = SVC(kernel='linear')  # Support Vector Classifier
svc.fit(x_train, y_train)

#Todos los parámetros restantes pueden verse en detalle aquí: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# PREDICCIÓN:

y_pred = svc.predict(x_test)
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)

#Podemos obtener la precisión del modelo obtenido:
score = svc.score(x_test, y_test)
print("Precisión: ", score)

#Igualmente podemos calcular la matriz de confusión generada por el modelo y el conjunto de validación:
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# PROBAR CON OTRAS FUNCIONES KERNEL:
from sklearn.svm import SVC

svc2 = SVC(kernel='sigmoid')  # Support Vector Classifier
svc2.fit(x_train, y_train)

score = svc2.score(x_test, y_test)
print("Precisión: ", score)


from sklearn.svm import SVC

svc2 = SVC(kernel='poly', degree=3)  # Support Vector Classifier
svc2.fit(x_train, y_train)

score = svc2.score(x_test, y_test)
print("Precisión: ", score)

from sklearn.svm import SVC

svc2 = SVC(kernel='rbf')  # Support Vector Classifier
svc2.fit(x_train, y_train)

score = svc2.score(x_test, y_test)
print("Precisión: ", score)


# DIBUJAR HIPERPLANO Y VECTORES SOPORTE:

#Para ver una representación del hiperplano calculado y de los vectores de soporte obtenidos vamos a trabajar con otro set de datos que nos facilite su representación (puesto que con el de animales y especies la mayoría de variables explicativas son binarias, por lo que los puntos solo toman dos valores 0 o 1).
#En este ejemplo vamos a trabajar con un set de datos de vinos y un conjunto de cualidades medidas en ellos.
#Como variable respuesta tenemos una calidad de vino, puntuada entre 0 y 10.

import pandas as pd
df = pd.read_csv("winequality-red.csv", sep=",")
print(df)

#Para la representación gráfica del modelo SVM, vamos a quedarnos con los valores de la clase (vinos con calidad 3 y calidad 8).
#Para poder representarlo en un plano consideraremos dos variables explicativas: el pH y el alcohol
x_cols = ["pH", "alcohol"]
y_col = "quality"

df = df[df[y_col].isin([3,8])]

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
X = df[x_cols].values
y = df[y_col].values

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.ylabel('alcohol')
plt.xlabel('pH')
plt.show()

## FIN



