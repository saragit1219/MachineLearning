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

# CONJUNTO DE ENTRENAMIENTO Y VALIDACIÓN:

from sklearn.model_selection import train_test_split

x = df[x_cols].values
y = df["class_type"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# ÁRBOL DE DECISIÓN:
#Calculamos el modelo utilizando el componente 'DecisionTreeClassifier' de la librería sklearn

from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()
dectree.fit(x_train,y_train)

#Vemos que por defecto se utiliza el criterio de la impureza de Gini para calcular la significancia de las variables explicativas. Podríamos cambiarlo al de la entropía (criterion='entropy').
#Igualmente podríamos forzar una profundidad máxima de nuestro árbol con 'max_depth'. Por defecto a None, iterará hasta que las hojas pertenezcan todas a una clase determinada (sean puras).
#Otra configuración interesante es 'min_impurity_decrease', la cual no permitiría descender en el árbol si no se reduce la impureza al menos el umbral aquí configurado.
#Todos los parámetros restantes pueden verse en detalle aquí: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html


# PREDICCIÓN:
#Utilizamos ahora el clasificador obtenido para predecir sobre el conjunto de validación

y_pred = dectree.predict(x_test)
#Podemos visualizar la equivalencia entre datos, clase real y clase predicha en forma de dataframe:
df_result = pd.DataFrame(x_test, columns=x_cols)
df_result["real_class"] = y_test
df_result["predicted_class"] = y_pred
print(df_result)


#Podemos obtener la precisión del modelo obtenido:
from sklearn import metrics

print("Precisión:",metrics.accuracy_score(y_test, y_pred))

#REPRESENTACIÓN DEL ÁRBOL OBTENIDO:

from sklearn import tree
import matplotlib.pyplot as plt

plot_size = 20
fig, ax = plt.subplots(figsize=(plot_size, plot_size))
tree.plot_tree(dectree, ax=ax, filled=True, feature_names=x_cols, class_names=dectree.classes_)
plt.show()


#En cada nodo se representa:

# - condición lógica a evaluar en el nodo (ej: milk <= 0.5)
# - gini: El valor de la impureza del nodo. Es decir, a esa altura determinada del árbol, que probabilidad hay de realizar una clasificación incorrecta.
# - samples: Número de muestras que han llegado a esa altura del árbol (el resto ya han sido previamente clasificadas por otras ramas)
# - values: array que representa el número de muestras para cada clase que hay a esa altura del árbol. Cada clase se representa como una posición del array.
# - class: clase final que se le otorga a una muestra a dicha altura del árbol.




## FIN




