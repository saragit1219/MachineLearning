import pandas as pd

df = pd.read_csv("real_estate.csv", sep=",")
print(df)

# Alguna información estadística:
rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 1) # Menos la variable respuesta

print("\nResumen estadístico de las variables:")
print("\n")
print(df.describe())

# Para este ejercicio vamos a intentar predecir la variable respuesta "precio" frente a las dos variables explicativas "edad de la casa" y "cercanía a la estación más cercana"

#Para poder representar bien los datos y que una variable no tenga mayor peso que la otra, vamos a normalizar las dos variables explicativas (puesto que las escalas de ambas unidades son muy diferentes: la distancia se mide en metros, por lo que es del orden de miles, mientras que los años suelen ser unidades bajas)

df['X3 distance to the nearest MRT station'] *= (1.0/df['X3 distance to the nearest MRT station'].max())
df['X2 house age'] *= (1.0/df['X2 house age'].max())

x = df[['X3 distance to the nearest MRT station', 'X2 house age']].values
y = df['Y house price of unit area'].values.reshape(-1,1)

# Usamos la función 'train_test_split' de sklearn para dividir nuestro conjunto de datos en dos subconjuntos: entrenamiento vs. validación (puesto que estamos realizando un aprendizaje supervisado).
# Utilizamos una proporción 80% de datos para entrenamiento y 20% para validación.
# El parámetro 'random_state' nos permite fijar una semilla en el procedimiento de separación de los datos para eliminar el factor aleatorio en la división (de cara a poder repetir la misma separación para la explicación del curso, pero en un caso real debería ser una separación aleatoria).

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Podemos representar visualmente los subconjuntos obtenidos:

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Conjunto de entrenamiento
df_train = pd.DataFrame({'distance': list(map(lambda t: t[0], x_train)),
                         'age': list(map(lambda t: t[1], x_train)),
                         'price': y_train.flatten()})

print(df_train)

threedee = plt.figure().gca(projection='3d')

threedee.scatter(df_train['distance'],
                 df_train['age'],
                 df_train['price'], color='green')

threedee.set_xlabel('Cercanía')
threedee.set_ylabel('Años')
threedee.set_zlabel('Precio')
plt.show()

# Conjunto de validación

df_test = pd.DataFrame({'distance': list(map(lambda t: t[0], x_test)),
                        'age': list(map(lambda t: t[1], x_test)),
                        'price': y_test.flatten()})

print(df_test)

threedee = plt.figure().gca(projection='3d')

threedee.scatter(df_test['distance'],
                 df_test['age'],
                 df_test['price'], color='purple')

threedee.set_xlabel('Cercanía')
threedee.set_ylabel('Años')
threedee.set_zlabel('Precio')
plt.show()


## REGRESIÓN LINEAL MÚLTIPLE:

#Una vez aclaradas las variables explicativas y la respuesta, y obtenidos los subconjuntos de entrenamiento y validación, modelamos nuestra regresión lineal múltiple.
#Calculamos el modelo utilizando el componente 'LinearRegression' de la librería sklearn.linear_model (al igual que con la regresión lineal simple, salvo que esta vez el vector X tendrá dos dimensiones, las dos variables explicativas).

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Podemos acceder a los coeficientes del plano generado por el modelo:
intercepto = regressor.intercept_[0]
gradientes = regressor.coef_[0] # Habrá dos puesto que hemos utilizado dos variables explicativas

print("Intercepto: ", intercepto)
print("Gradientes: ", gradientes)

print("Ecuación plano: y = {} + {} x1 + {} x2".format(intercepto, gradientes[0], gradientes[1]))

#Y podemos representarla gráficamente:

import matplotlib.pyplot as pl
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

# Representa los puntos de entrenamiento
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_train['distance'],
                 df_train['age'],
                 df_train['price'], color='green')
ax.set_xlabel("cercanía")
ax.set_ylabel("años")
ax.set_zlabel("precio")

# Representa el plano de la regresión múltiple
coefs = regressor.coef_
intercept = regressor.intercept_

xs = np.tile(np.arange(2), (2, 1))
ys = np.tile(np.arange(2), (2, 1)).T
zs = xs*coefs[0][0]+ys*coefs[0][1]+intercept

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()

## PREDICCIÓN:
y_pred = regressor.predict(x_test)
df_predicted = pd.DataFrame({'distance': list(map(lambda t: t[0], x_test)),
                             'age': list(map(lambda t: t[1], x_test)),
                             'actual price': y_test.flatten(),
                             'predicted price': y_pred.flatten()})
print(df_predicted)

#Podemos representar visualmente esta predicción, viendo gráficamente como los puntos predichos se ajustan al plano calculado:

import matplotlib.pyplot as pl
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

# Representa los puntos predichos
fig = pl.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_predicted['distance'],
           df_predicted['age'],
           df_predicted['predicted price'], color='orange')
ax.set_xlabel("cercanía")
ax.set_ylabel("años")
ax.set_zlabel("precio")

# Representa el plano de la regresión múltiple
coefs = regressor.coef_
intercept = regressor.intercept_

xs = np.tile(np.arange(2), (2, 1))
ys = np.tile(np.arange(2), (2, 1)).T
zs = xs*coefs[0][0]+ys*coefs[0][1]+intercept

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()

#Igualmente podemos representar gráficamente como los errores cometidos con respecto a los salarios reales. Lo haremos sobre una muestra de 20 puntos de validación para que quepan en la gráfica:
df_predicted[["actual price", "predicted price"]].head(20).plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

## EVALUACIÓN DEL MODELO:
#Aparte de la evaluación visual del modelo, podemos obtener los errores ya vistos en el módulo para verificar la bondad de la regresión:
from sklearn import metrics
import numpy as np
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))

#En este caso por ejemplo podemos utilizar el "Root Mean Squared Error" como medida del error con respecto a los datos reales, puesto que se expresa en las mismas unidades que la variable a predecir (en este caso, el salario). Vemos que de media existe un error de 8.86M dólares, lo que supone un 23.31% con respecto a la media de precios que existen en los datos.
#Un coeficiente de determinación de 0.55 indica que es un modelo recoge aproximadamente la mitad de la variabilidad de los datos origen (por lo que no es muy bueno).


## FIN ##







