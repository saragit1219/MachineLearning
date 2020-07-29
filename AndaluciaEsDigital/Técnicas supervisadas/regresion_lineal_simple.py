import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("salary_data.csv", sep=",")
print(df)

# Alguna información estadística:
rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols - 1) # Menos la variable respuesta

print("\nResumen estadístico de las variables:")
print("\n")
print(df.describe())

# Usamos la función 'train_test_split' de sklearn para dividir nuestro conjunto de datos en dos subconjuntos: entrenamiento vs. validación (puesto que estamos realizando un aprendizaje supervisado).
# Utilizamos una proporción 80% de datos para entrenamiento y 20% para validación.
# El parámetro 'random_state' nos permite fijar una semilla en el procedimiento de separación de los datos para eliminar el factor aleatorio en la división (de cara a poder repetir la misma separación para la explicación del curso, pero en un caso real debería ser una separación aleatoria).

from sklearn.model_selection import train_test_split

x = df['YearsExperience'].values
y = df['Salary'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Podemos representar visualmente los subconjuntos obtenidos:
# Conjunto de entrenamiento

df_train = pd.DataFrame({'YearsExperience': x_train, 'Salary': y_train})

print(df_train)

traintestplot = plt.figure().gca()

traintestplot.scatter(df_train['YearsExperience'], df_train['Salary'], color='green')

traintestplot.set_xlabel('YearsExperience')
traintestplot.set_ylabel('Salary')
plt.show()

# Conjunto de validación

df_test = pd.DataFrame({'YearsExperience': x_test, 'Salary': y_test})

print(df_test)

traintestplot = plt.figure().gca()

traintestplot.scatter(df_test['YearsExperience'], df_test['Salary'], color='purple')

traintestplot.set_xlabel('YearsExperience')
traintestplot.set_ylabel('Salary')
plt.show()


#Una vez aclaradas las variables explicativa (en este caso la única que hay ya que es simple) y la respuesta, modelamos nuestra regresión lineal.
#Calculamos el modelo utilizando el componente 'LinearRegression' de la librería sklearn.linear_model.
#Necesitamos trasponer los vectores x e y para poder pasárselos al modelo (en forma columnar en vez de fila). Para ello usamos la función reshape de NumPy.

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

#Podemos acceder a los coeficientes de la linea recta generada por el modelo:
intercepto = regressor.intercept_[0]
gradiente = regressor.coef_[0][0] # Sólo hay uno puesto que solo tenemos una variable explicativa

print("Intercepto: ", intercepto)
print("Gradiente: ", gradiente)
print("Ecuación recta: y = {} + {} x".format(intercepto, gradiente))

# Y podemos representarla gráficamente:
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, regressor.predict(x_train.reshape(-1, 1)), color='red', linewidth=2)
plt.show()

## PREDICCION:
#Utilizamos ahora el modelo de regresión obtenido para predecir sobre el conjunto de validación
y_pred = regressor.predict(x_test.reshape(-1, 1))
df_predicted = pd.DataFrame({'Salario real': y_test.flatten(), 'Salario predicho': y_pred.flatten()})
print(df_predicted)

# Podemos representar visualmente esta predicción, así como los errores cometidos con respecto a los salarios reales:
plt.scatter(x_test, y_test, color='purple')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()

df_predicted.plot(kind='bar')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# EVALUACIÓN DEL MODELO:
#Aparte de la evaluación visual del modelo, podemos obtener los errores ya vistos en el módulo para verificar la bondad de la regresión:
from sklearn import metrics
import numpy as np
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R^2:', metrics.r2_score(y_test, y_pred))

# En este caso por ejemplo podemos utilizar el "Root Mean Squared Error" como medida del error con respecto a los datos reales, puesto que se expresa en las mismas unidades que la variable a predecir (en este caso, el salario). Vemos que de media existe un error de 3580 dólares, lo que supone un 4.73% con respecto a la media de salarios que existen en los datos.
# Igualmente, un coeficiente de determinación de 0.988 indica que el modelo explica una gran variabilidad de los datos de origen.




## FIN ##