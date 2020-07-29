# EJ 3 ÁRBOL DE DECISIÖN # Sara Barrera Romero

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


# ÁRBOL DE DECISIÓN:

from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(x_train,y_train)

# PREDICCIÓN:
# Podemos hacer una predicción:
y_pred = dectree.predict(x_test)
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)

#Precisión del modelo obtenido:
from sklearn import metrics

print("Precisión:",metrics.accuracy_score(y_test, y_pred))


#REPRESENTACIÓN DEL ÁRBOL OBTENIDO:

from sklearn import tree
import matplotlib.pyplot as plt

plot_size = 11
font_size = 4.9
fig, ax = plt.subplots(figsize=(plot_size, plot_size))
tree.plot_tree(dectree, ax=ax, filled=True, feature_names=x_cols, class_names=dectree.classes_, fontsize=font_size)
plt.show()


# Para los datos de la nueva seta que se plantea en el ejercicio me sale que su class = p, es decir, venenosa.