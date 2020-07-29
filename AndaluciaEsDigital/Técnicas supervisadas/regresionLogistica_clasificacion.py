import pandas as pd

#En este caso no vamos a predecir la variable especie sino la variable 'fins' (aletas) para que sea una variable respuesta binaria.
#Para ello podemos valernos de todas las variables explicativas disponibles.
#Todas son binarias (no/si, codificadas como 0/1) menos la variable "legs" que es numérica.

df = pd.read_csv("zoo.csv", sep=",")
print(df)

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 2) # Menos el nombre del animal y la variable respuesta

print("\nTotal de valores posibles para la variable respuesta 'fins': ", df["fins"].unique())
print("\n")
print(df.describe())

#Guardamos en una variable 'x_list' todas las columnas que representan variables explicativas.
#En nuestro caso, son todas las columnas del dataframe menos la variable respuesta 'class_type' y la variable 'animal_name', que indica el nombre del animal (único para cada observación)

x_cols = list(df.columns)
x_cols.remove("animal_name")
x_cols.remove("class_type")
x_cols.remove("fins")



# CONJUNTO DE ENTRENAMIENTO Y VALIDACIÓN:

#Usamos la función 'train_test_split' de sklearn para dividir nuestro conjunto de datos en dos subconjuntos: entrenamiento vs. validación (puesto que estamos realizando un aprendizaje supervisado).
#Utilizamos una proporción 80% de datos para entrenamiento y 20% para validación.
#El parámetro 'random_state' nos permite fijar una semilla en el procedimiento de separación de los datos para eliminar el factor aleatorio en la división (de cara a poder repetir la misma separación para la explicación del curso, pero en un caso real debería ser una separación aleatoria).

from sklearn.model_selection import train_test_split

x = df[x_cols].values
y = df["fins"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# REGRESIÓN LOGÍSTICA

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

#El clasificador de regresión logística funciona tanto con clases binarias como multiclase.
#El parámetro 'multi_class' permite determinar este comportamiento. Por defecto 'auto' se detercará automaticamente. Si especificamos 'ovr' se ajustará un problema binario para cada variable respuesta. Si es 'multinomial' se tratará como un multiclase.
#Otro parámetro interesante puede ser 'max_iter', para fijar un número máximo de iteraciones al intentar ir reduciendo el coste del modelo.
#Todos los parámetros restantes pueden verse en detalle aquí: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


# PREDICCION:

y_pred = logreg.predict(x_test)
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_fins"] = y_test
df_result["predicted_fins"] = y_pred
print(df_result)

#Podemos obtener la precisión del modelo obtenido:
score = logreg.score(x_test, y_test)
print("Precisión: ", score)

#Igualmente podemos calcular la matriz de confusión generada por el modelo y el conjunto de validación:
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# REPRESENTACIÓN DE LA FUNCIÓN SIGMOIDE:
#Podemos representar la funcion sigmoide obtenida por nuestro modelo de regresión logística

print(__doc__)


# Code source: Gael Varoquaux
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from scipy.special import expit

# General a toy dataset:s it's just a straight line with some Gaussian noise:
X = df[["legs"]].values
y = df["fins"].values

# Fit the classifier
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

# and plot the result
plt.figure(1, figsize=(7, 5))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

ols = linear_model.LinearRegression()
ols.fit(X, y)
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('fins')
plt.xlabel('legs')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
plt.tight_layout()
plt.show()

## FIN




