# EJ 3 REGRESIÓN LOGÍSTICA # Sara Barrera Romero

# NOTA: Vamos a usar en la clasificación de los datos de entrenamiento y validación random_state=0 para que en los 4 pasos de el ejercicio 3 tengamos
# la misma clasificación de los datos de entrenamiento y validación.

import pandas as pd

# Queremos predecir si una seta es venenosa (variable respuesta binaria) o no a partir de una serie de datos:

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

# REGRESIÓN LOGÍSTICA

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=500)
logreg.fit(x_train, y_train)


# PREDICCION:

y_pred = logreg.predict(x_test)
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)


#Precisión del modelo obtenido:
score = logreg.score(x_test, y_test)
print("Precisión: ", score)


#Matriz de confusión generada por el modelo y el conjunto de validación:
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))




