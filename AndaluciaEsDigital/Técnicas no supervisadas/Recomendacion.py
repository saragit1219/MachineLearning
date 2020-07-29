import pandas as pd


#Para el ejemplo del recomendador vamos a trabajar con un catálogo de libros de forma que recomendemos libros en base a la similitud entre ellos (recomendador basado en contenido).
df = pd.read_csv("books.csv", sep=",")
print(df)

rows, cols = df.shape
print("\nFilas: ", rows)
print("Variables explicativas: ", cols)
print("Columnas: ", df.columns)

print(df.describe())

"""
En este caso, para poder hablar de similitud entre libros, vamos a quedarnos con cuatro variables explicativas:

authors: Supondremos que al usuario le gustarán libros escritos por el mismo autor
original_publication_year: Supondremos que al usuario le gustarán libros escritos en la misma época
original_title: Las palabras clave de los títulos pueden dar pistas de las preferencias del usuario
language_code: Supondremos que al usuario le gustarán libros escritos en el mismo idioma
"""

features = ["authors", "original_publication_year", "original_title", "language_code"]

# ASIGNACIÓN DE PALABRAS CLAVES:

#Para cada observación vamos a añadir una nueva columna con la combinación de todas las palabras clave establecidas anteriormente, que serán la base de las recomendaciones:

def get_keywords(row):
    text = ""
    for feature in features:
        text += str(row[feature])+" "
    return text

df["keywords"] = df.apply(get_keywords, axis=1)

#Mostramos el df con la nueva columna:

df[["id", "original_title", "keywords"]]
print(df)


# MATRIZ DE PRODUCTOS:

"""
A continuación tenemos que calcular cual es la similitud entre los distintos libros.

Para ello, vamos a computar el número de ocurrencias de cada palabra clave que hemos extraido de los libros (formada por año de publicación, autor, título e idioma) en el total del dataset de los libros.

Utilizamos para ello el componente 'CountVectorizer' del modulo sklearn.feature_extraction.text
"""


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["keywords"])


#Del objeto CountVectorized podemos obtener tanto las palabras clave extraidas como los conteos de las ocurrencias

print(cv.get_feature_names()[:10], "...")
print(count_matrix.toarray())

"""
Finalmente, para representar la similitud entre los libros, utilizaremos la distancia "cosine_similarity" del modulo sklearn.metrics.pairwise.

El resultado de esta función es una matriz simétrica en la que cada fila/columna representa una observación (un libro). Cada posición de la matriz indicará cuanto de similar (en base a las keywords extraidas) es cada par de libros, con un valor entre 0 y 1. La diagonal de dicha matriz por tanto será de 1, puesto que representa la similitud de las palabras clave de un libro consigo mismo (matriz simétrica).
"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix)
print(cosine_sim)


# OBTENCIÓN DE PRODUCTOS SIMILARES:

#Finalmente, construimos una serie de funciones que nos ayuden a buscar dentro de la matriz de "cosine_similarity" los libros más similares (en cuanto a las keywords escogidas) con respecto a un libro de referencia:

# Dado el id de un libro, retornamos el título
def get_title_from_id(ID):
    return df[df.id == ID]["original_title"].values[0]


# Dado el título de un libro, retornamos el id
def get_id_from_title(title):
    return df[df.original_title == title]["id"].values[0]


# Búsqueda de los libros más similares
def get_top_N_recommendations_for(book_title, n=5):
    title_index = get_id_from_title(book_title)

    # Recuperamos de la matriz de similitudes-coseno el índice correspondiente al titulo recibido
    similar_books = list(enumerate(cosine_sim[title_index]))

    # Ordenamos la fila por similitud
    sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:]

    # Imprimimos el top N de libros más similares
    print("Top {} de libros similares a '{}' son:\n".format(n, book_title))
    for i in range(0, n):
        print(" [{}]: {}".format(i + 1, get_title_from_id(sorted_similar_books[i][0])))

#Probamos con algunos ejemplos:

get_top_N_recommendations_for("Harry Potter and the Philosopher's Stone", 5)

get_top_N_recommendations_for("The Godfather")


## FIN

