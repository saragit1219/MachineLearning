# EJERCICIO 2 (REGRESIÓN). SARA BARRERA ROMERO.

import pandas as pd

df = pd.read_csv("melbourne_housing.csv", sep="|")
print(df)

import seaborn as sns
import matplotlib.pyplot as plt

df_pairplot = df[['Price', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']]
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df_pairplot)
plt.show()

# En mi opinión creo que la variable que guarda una mayor dependencia lineal con la variable respuesta precio es: "BuildingArea".
# Por tanto, vamos a modelar una regresión lineal simple con esta variable para predecir la variable respuesta precio:

# Voy a suponer que los valores negativos de la columna 'BuildingArea' son errores, por tanto, los voy a eliminar:
# (como usted me dijo, esto solo supone eliminar en torno a un 10%, por lo que no es ningún problema eliminarlos)
df = df.drop(df[df['BuildingArea']<0].index)


# Información estadística:
rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 1) # Menos la variable respuesta

print("\nResumen estadístico de las variables:")
print("\n")
print(df.describe())


# Vamos a utilizar la función 'train_test_split' de sklearn para dividir el conjunto de datos en dos: entrenamiento y validación.
# Utilizamos como en la teoría 80% para el entrenamiento y 20% para la validación.

from sklearn.model_selection import train_test_split

x = df['BuildingArea'].values
y = df['Price'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Visualizamos el conjunto de entrenamiento:

df_train = pd.DataFrame({'BuildingArea': x_train, 'Price': y_train})

traintestplot = plt.figure().gca()
traintestplot.scatter(df_train['BuildingArea'], df_train['Price'], color='green')
traintestplot.set_xlabel('BuildingArea')
traintestplot.set_ylabel('Price')
plt.show()

# Visualizemos el conjunto de validación:

df_test = pd.DataFrame({'BuildingArea': x_test, 'Price': y_test})

traintestplot = plt.figure().gca()
traintestplot.scatter(df_test['BuildingArea'], df_test['Price'], color='purple')
traintestplot.set_xlabel('BuildingArea')
traintestplot.set_ylabel('Price')
plt.show()

# MODELAR LA REGRESIÓN LINEAL SIMPLE:

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

# La ecuación de la línea recta resultante tiene como intercepto y gradiente:
intercepto = regressor.intercept_[0]
gradiente = regressor.coef_[0][0] # Sólo hay uno puesto que solo tenemos una variable explicativa

print("Intercepto: ", intercepto)
print("Gradiente: ", gradiente)
print("Ecuación recta: y = {} + {} x".format(intercepto, gradiente))

# Y podemos representarla gráficamente la línea recta resultante sobre el conjunto de entrenamiento de los datos:
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, regressor.predict(x_train.reshape(-1, 1)), color='red', linewidth=2)
plt.show()


## PREDICCIÓN:

# Vamos a utilizar el modelo para hacer predicciones sobre el conjunto de datos de validación:

y_pred = regressor.predict(x_test.reshape(-1, 1))
df_predicted = pd.DataFrame({'Precio real': y_test.flatten(), 'Precio predicho': y_pred.flatten()})

# Vamos a representar visualmente esta predicción:
plt.scatter(x_test, y_test, color='purple')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

#Podemos representar gráficamente los errores cometidos con respecto al precio real. Lo haremos con 20 puntos como en la teoría:
df_predicted[['Precio real', 'Precio predicho']].head(20).plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

## EVALUACIÓN DEL MODELO:

from sklearn import metrics
import numpy as np
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))

# Podemos fijarnos en 'Root Mean Squared Error' (que se expresa en las mismas unidades que nuestra variable respuesta) para ver que existe
# un error de 369796.64 dólares Australianos, lo que supone aproximadamente un 34,8% con respecto a la media de precios que existen en los datos.
# El coeficiente de determinación de 0.604 indica que el modelo explica aproximadamente un 60% de los datos de origen.
# (Aunque estos datos varían un poco según la semilla que tome el programa)


# Para en punto 4 de la tarea, en mi opinión, la segunda variable explicativa de mayor dependencia lineal con la variable precio es 'Distance'.
# Vamos a intentar predecir la variable respuesta 'Price' frente a las dos variables explicativas 'BuildingArea' y 'Distance'.

# Al igual que antes, voy a suponer que los valores negativos de la columna 'Distance' son errores, por tanto, los voy a eliminar:
df = df.drop(df[df['Distance']<0].index)

# Vamos a normalizar las dos variables explicativas:
df['BuildingArea'] *= (1.0/df['BuildingArea'].max())
df['Distance'] *= (1.0/df['Distance'].max())

x = df[['BuildingArea', 'Distance']].values
y = df['Price'].values.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Vamos a representar visualmente los subconjuntos obtenidos:
from mpl_toolkits.mplot3d import Axes3D

# Conjunto de entrenamiento:
df_train = pd.DataFrame({'BuildingArea': list(map(lambda t: t[0], x_train)),
                         'Distance': list(map(lambda t: t[1], x_train)),
                         'Price': y_train.flatten()})
threedee = plt.figure().gca(projection='3d')

threedee.scatter(df_train['BuildingArea'],
                 df_train['Distance'],
                 df_train['Price'], color='green')

threedee.set_xlabel('Área de construcción')
threedee.set_ylabel('Distancia')
threedee.set_zlabel('Precio')
plt.show()


# Conjunto de validación

df_test = pd.DataFrame({'BuildingArea': list(map(lambda t: t[0], x_test)),
                        'Distance': list(map(lambda t: t[1], x_test)),
                        'Price': y_test.flatten()})

print(df_test)

threedee = plt.figure().gca(projection='3d')

threedee.scatter(df_test['BuildingArea'],
                 df_test['Distance'],
                 df_test['Price'], color='purple')

threedee.set_xlabel('Área de construcción')
threedee.set_ylabel('Distancia')
threedee.set_zlabel('Precio')
plt.show()

## REGRESIÓN LINEAL MÚLTIPLE:

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Coeficientes del plano generado por el modelo:
intercepto = regressor.intercept_[0]
gradientes = regressor.coef_[0] # Hay dos puesto que hemos utilizado dos variables explicativas

print("Intercepto: ", intercepto)
print("Gradientes: ", gradientes)

print("Ecuación plano: y = {} + {} x1 + {} x2".format(intercepto, gradientes[0], gradientes[1]))

# Veámoslo gráficamente:

# Representa los puntos de entrenamiento
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_train['BuildingArea'],
                 df_train['Distance'],
                 df_train['Price'], color='green')
ax.set_xlabel('Área de construcción')
ax.set_ylabel('Distancia')
ax.set_zlabel('Precio')

# Plano de la regresión múltiple
coefs = regressor.coef_
intercept = regressor.intercept_

xs = np.tile(np.arange(2), (2, 1))
ys = np.tile(np.arange(2), (2, 1)).T
zs = xs*coefs[0][0]+ys*coefs[0][1]+intercept

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()

# Predicción:

y_pred = regressor.predict(x_test)
df_predicted = pd.DataFrame({'BuildingArea': list(map(lambda t: t[0], x_test)),
                             'Distance': list(map(lambda t: t[1], x_test)),
                             'actual price': y_test.flatten(),
                             'predicted price': y_pred.flatten()})
print(df_predicted)

# Vemos esta predicción visualmente:

# Representa los puntos predichos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_predicted['BuildingArea'],
           df_predicted['Distance'],
           df_predicted['predicted price'], color='orange')
ax.set_xlabel('Área de construcción')
ax.set_ylabel('Distancia')
ax.set_zlabel('Precio')

# Representa el plano de la regresión múltiple
coefs = regressor.coef_
intercept = regressor.intercept_

xs = np.tile(np.arange(2), (2, 1))
ys = np.tile(np.arange(2), (2, 1)).T
zs = xs*coefs[0][0]+ys*coefs[0][1]+intercept

ax.plot_surface(xs,ys,zs, alpha=0.5)
plt.show()

# De nuevo podemos representar gráficamente los errores cometidos con respecto al precio real:
df_predicted[["actual price", "predicted price"]].head(20).plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Y, por último, hacemos la EVALUACIÓN DEL MODELO:

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))

# El coeficiente de determinación de aproximadamente 0.6 nos indica que el modelo tiene una efectividad del 60%.
# Y 'Root Mean Squared Error' = 354387.8279692509 indica que existe un error de aproximadamente 354387.8 dólares australianos, lo que supone un 33,3% con respecto a la media de precios que existen en los datos.
