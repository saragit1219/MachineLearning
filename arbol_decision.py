# Imports necesarios:

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

artists_billboard = pd.read_csv(r"artists_billboard_fix3.csv")

colfil = artists_billboard.shape    # (columnas, filas)
cabeza = artists_billboard.head()   # primeros registros

gettop = artists_billboard.groupby('top').size()    # Cuantos artistas consiguieron el top
#sb.catplot('top',data=artists_billboard,kind="count") # Cuantos artistas consiguieron el top (gráficamente)

######################################################
# VISUALIZAR LOS DATOS DE ENTRADA:
'''sb.catplot('artist_type',data=artists_billboard,kind="count") # género
sb.catplot('mood',data=artists_billboard,kind="count", aspect=3) #mood
sb.catplot('tempo',data=artists_billboard,hue='top',kind="count") #tempo/top
sb.catplot('genre',data=artists_billboard,kind="count", aspect=3) #género musical
sb.catplot('anioNacimiento',data=artists_billboard,kind="count", aspect=3) #año de nacimiento (año 0 representa desconocido)
plt.show()'''

######################################################
# BALANCEO DE LOS DATOS:
#comparamos los top y no-top según la duración de la canción y la fecha de chart
'''f1 = artists_billboard['chart_date'].values     #devuelve todas las fechas de cada canción
f2 = artists_billboard['durationSeg'].values    #devuelve la duración de cada canción

colores=['orange','blue'] #color de los puntos
tamanios=[20,100]  #tamaño de los puntos

asignar=[]
asignar2=[]
# para cada índice, fila:
for index, row in artists_billboard.iterrows():
    asignar.append(colores[row['top']])     #asigna color naranja:0 (no-top), azul:1 (top)
    asignar2.append(tamanios[row['top']])   #asigna 60:0 (no-top), 40:1 (top)

plt.scatter(f1, f2, c=asignar, s=asignar2)
plt.axis([20030101,20160101,0,700])
# en el eje X se pinta el año de la canción (2003-2016) y en el eje Y se pinta la duración de la canción (en segundos). Se aprecia que
# la mayoría de los top1 están entre los años 2003 y 2015.
plt.show()'''


# Veamos si hay alguna reñación entre el año de nacimiento y la duración de la canción
'''colores=['orange','blue']   # color de los puntos

f1 = artists_billboard['anioNacimiento'].values
f2 = artists_billboard['durationSeg'].values

asignar=[]
for index, row in artists_billboard.iterrows():
    asignar.append(colores[row['top']])     # naranja:0, azul:1

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([1960,2005,0,600])
# No perece haber ningún patron a la vista, están bastante mezclados los top de los no-top.
plt.show()'''


######################################################
## PREPARAMOS LOS DATOS:

#Vamos a arreglar el problema de los años de nacimiento que están a cero. Realmente la característica que queremos
# obtener es: Sabiendo el año de nacimiento del cantante calcular la edad que tenía en el momento de aparecer el billboard.
# Ej: si nació en 1982 y su canción apareció en 2012, entonces tenía 30 años.

def edad_fix(anio):
    """Reemplaza el valor 0 por None"""
    if anio==0:
        return None
    return anio

#cambiamos los años de nacimiento a 0 por None:
artists_billboard['anioNacimiento']=artists_billboard.apply(lambda x: edad_fix(x['anioNacimiento']), axis=1);

# Vamos a crear una nueva columna <edad_en_billboard> donde añadimos la edad del artista en el billboard restando al año del billboard
# el año de nacimiento. Tendremos None en la fila donde el año de nacimiento sea None.

def calcula_edad(anio,cuando):
    """Calcula la edad del cantante en el billboard"""
    cad = str(cuando)
    momento = cad[:4]   #el año son los primero 4 dígitos de la fecha
    if anio==0.0:
        return None
    return int(momento) - anio

artists_billboard['edad_en_billboard']=artists_billboard.apply(lambda x: calcula_edad(x['anioNacimiento'],x['chart_date']), axis=1);

# Finalmente añadimos edades aleatorias a los registros que nos faltan: Para ello, obtenemos el promedio (avg)
# de la edad de nuestro conjunto y su desvío estándar(std) (por ello queriamos los Nones) y pedimos valores random que van
# desde [avg-std, avg+std].

age_avg = artists_billboard['edad_en_billboard'].mean()     #media
age_std = artists_billboard['edad_en_billboard'].std()      #desviación
age_null_count = artists_billboard['edad_en_billboard'].isnull().sum()  # número de registros con None en edad_en_billboard
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)  #lista de age_null_count registros de edades aleatorias en el intervalo descrito

conValoresNulos = np.isnan(artists_billboard['edad_en_billboard']) #para cada fila devuelve True o False (True:None, False:no None)
# meter los valores random en los Nones de la tabla:
artists_billboard.loc[np.isnan(artists_billboard['edad_en_billboard']), 'edad_en_billboard'] = age_null_random_list
artists_billboard['edad_en_billboard'] = artists_billboard['edad_en_billboard'].astype(int)

'''print("Edad Promedio: " + str(age_avg))
print("Desvió Std Edad: " + str(age_std))
print("Intervalo para asignar edad aleatoria: " + str(int(age_avg - age_std)) + " a " + str(int(age_avg + age_std)))

# Podemos visualizar los valores que hemos incluido en verde en el siguiente gráfico:
f1 = artists_billboard['edad_en_billboard'].values
f2 = artists_billboard.index

colores = ['orange','blue','green']

asignar=[]
for index, row in artists_billboard.iterrows():
    if (conValoresNulos[index]):
        asignar.append(colores[2]) # verde
    else:
        asignar.append(colores[row['top']])

plt.scatter(f1, f2, c=asignar, s=30)
plt.axis([15,50,0,650])
plt.show()'''


######################################################
### MAPEO DE LOS DATOS:

# Transformamos varios datos de entrada en valores categóricos:

# Mood Mapping
artists_billboard['moodEncoded'] = artists_billboard['mood'].map( {'Energizing': 6,
                                        'Empowering': 6,
                                        'Cool': 5,
                                        'Yearning': 4, # anhelo, deseo, ansia
                                        'Excited': 5, #emocionado
                                        'Defiant': 3,
                                        'Sensual': 2,
                                        'Gritty': 3, #coraje
                                        'Sophisticated': 4,
                                        'Aggressive': 4, # provocativo
                                        'Fiery': 4, #caracter fuerte
                                        'Urgent': 3,
                                        'Rowdy': 4, #ruidoso alboroto
                                        'Sentimental': 4,
                                        'Easygoing': 1, # sencillo
                                        'Melancholy': 4,
                                        'Romantic': 2,
                                        'Peaceful': 1,
                                        'Brooding': 4, # melancolico
                                        'Upbeat': 5, #optimista alegre
                                        'Stirring': 5, #emocionante
                                        'Lively': 5, #animado
                                        'Other': 0,'':0} ).astype(int)
# Tempo Mapping
artists_billboard['tempoEncoded'] = artists_billboard['tempo'].map( {'Fast Tempo': 0, 'Medium Tempo': 2, 'Slow Tempo': 1, '': 0} ).astype(int)
# Genre Mapping
artists_billboard['genreEncoded'] = artists_billboard['genre'].map( {'Urban': 4,
                                          'Pop': 3,
                                          'Traditional': 2,
                                          'Alternative & Punk': 1,
                                         'Electronica': 1,
                                          'Rock': 1,
                                          'Soundtrack': 0,
                                          'Jazz': 0,
                                          'Other':0,'':0}
                                       ).astype(int)
# artist_type Mapping
artists_billboard['artist_typeEncoded'] = artists_billboard['artist_type'].map( {'Female': 2, 'Male': 3, 'Mixed': 1, '': 0} ).astype(int)


# Mapping edad en la que llegaron al billboard
artists_billboard.loc[ artists_billboard['edad_en_billboard'] <= 21, 'edadEncoded']                         = 0
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 21) & (artists_billboard['edad_en_billboard'] <= 26), 'edadEncoded'] = 1
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 26) & (artists_billboard['edad_en_billboard'] <= 30), 'edadEncoded'] = 2
artists_billboard.loc[(artists_billboard['edad_en_billboard'] > 30) & (artists_billboard['edad_en_billboard'] <= 40), 'edadEncoded'] = 3
artists_billboard.loc[ artists_billboard['edad_en_billboard'] > 40, 'edadEncoded'] = 4

# Mapping Song Duration
artists_billboard.loc[ artists_billboard['durationSeg'] <= 150, 'durationEncoded']                          = 0
artists_billboard.loc[(artists_billboard['durationSeg'] > 150) & (artists_billboard['durationSeg'] <= 180), 'durationEncoded'] = 1
artists_billboard.loc[(artists_billboard['durationSeg'] > 180) & (artists_billboard['durationSeg'] <= 210), 'durationEncoded'] = 2
artists_billboard.loc[(artists_billboard['durationSeg'] > 210) & (artists_billboard['durationSeg'] <= 240), 'durationEncoded'] = 3
artists_billboard.loc[(artists_billboard['durationSeg'] > 240) & (artists_billboard['durationSeg'] <= 270), 'durationEncoded'] = 4
artists_billboard.loc[(artists_billboard['durationSeg'] > 270) & (artists_billboard['durationSeg'] <= 300), 'durationEncoded'] = 5
artists_billboard.loc[ artists_billboard['durationSeg'] > 300, 'durationEncoded'] = 6

# Finalmente obtenemos un nuevo conjunto de datos llamado <artists_encored> con el que tenemos los atributos definidos para crear el árbol.
# Para ello borramos todas las columnas que no necesitamos con drop.

# (ha creado una nueva columna para cada columna que ya teniamos con si encored correspondiente anteriormente definido y en la variable
# artists_encored ha borrado todas las anteriores para solo tener las encored) :
drop_elements = ['id','title','artist','mood','tempo','genre','artist_type','chart_date','anioNacimiento','durationSeg','edad_en_billboard']
artists_encoded = artists_billboard.drop(drop_elements, axis = 1)

# ANALIZAMOS NUESTROS DATOS DE ENTRADA CATEGÓRICOS:

artists_encoded.head() #primero datos de esta nueva tabla
artists_encoded.describe()  #datos estadísticos para cada columna

# Revisamos en las tablas como se reparten los top=1 en los diversos atributos mapeados. Sobre la columna sum están los top, que puede ser valor
# igual a 0 o a 1, solo se suma los que tienen igual a 1:
# (hacer una relación entre dos variables)
# Se puede ver que: La mayoría de top 1 los vemos en los estados de ánimo 5 y 6 con 46 y 43 canciones
artists_encoded[['moodEncoded', 'top']].groupby(['moodEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# Se puede ver que: Aqui están bastante repartidos, pero hay mayoría en tipo 3: artistas masculinos
artists_encoded[['artist_typeEncoded', 'top']].groupby(['artist_typeEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# Se puede ver que: Los géneros con mayoría son evidentemente los géneros 3 y 4 que corresponden con Urbano y Pop
artists_encoded[['genreEncoded', 'top']].groupby(['genreEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# Se puede ver que: El tempo con más canciones exitosas en el número 1 es el 2, tempo medio
artists_encoded[['tempoEncoded', 'top']].groupby(['tempoEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# Están bastante repartidos en relación a la duración de las canciones:
artists_encoded[['durationEncoded', 'top']].groupby(['durationEncoded'], as_index=False).agg(['mean', 'count', 'sum'])

# Edad con mayoría es la tipo 1 que comprende de 21 a 25 años:
artists_encoded[['edadEncoded', 'top']].groupby(['edadEncoded'], as_index=False).agg(['mean', 'count', 'sum'])


# Matriz con el coeficiente de correlación de Pearson (cada dos variables):
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sb.heatmap(artists_encoded.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#plt.show()


######################################################
## BUSCAMOS LA PROFUNDIDAD DEL ÁRBOL DE DECISIÓN:

# Antes de crear nuestro árbol vamos a buscar cuantos niveles de profundidad le vamos a asignar. Para ello usamos la función KFold
# que nos ayudará a crear varios subgrupos con nuestros datos de entrada para validad y valorar los árboles con diversos niveles de profundidad.
# De entre todos ellos cogeremos el de mejor resultado.

# Vamos a usar la librería : sklearn tree (buscamos un árbol de decisión, no de regresión)
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# Lo configuramos con los parámetros:
# criterion = entropy  (podría ser gini pero nuestras entradas con categóricas)
# min_samples_split = 20 (cantidad mínima de muestras que debe tener un nodo para subdividir)
# min_samples_leaf = 5 (cantidad mínima que puede tener una hoja final. Si tuviera menos, no se forma la hoja y se 'subiría' un nivel, su antecesor)
# class_weight = {1:3.5} (IMPORTANTÍSIMO: con esto descompensamos los desbalances que hubiera. En nuestro casi tenemos menos etiquetas tipo top=1
# por lo que le asignamos 3.5 al peso de la etiqueta 1 para compensar (el valor sale de dividir la cantidad de top=0, 494 con los top 1, 141.

# (estos valores se asignan a base de prueba y error muchas veces visualizando el árbol resultante)

cv = KFold(n_splits=10)  # Numero deseado de "folds" que haremos
accuracies = list()     # lista vacía
max_attributes = len(list(artists_encoded))     #número de cualidades que se estudian
depth_range = range(1, max_attributes + 1)  # depth = profundidad (rango de profundidad: 1 a 8)

# Testearemos la profundidad de 1 a cantidad de atributos +1
# (estudiamos el árbol para cada profundidad)
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                             min_samples_split=20,
                                             min_samples_leaf=5,
                                             max_depth=depth,
                                             class_weight={1: 3.5})
    for train_fold, valid_fold in cv.split(artists_encoded):
        f_train = artists_encoded.loc[train_fold]
        f_valid = artists_encoded.loc[valid_fold]

        model = tree_model.fit(X=f_train.drop(['top'], axis=1),
                               y=f_train["top"])
        valid_acc = model.score(X=f_valid.drop(['top'], axis=1),
                                y=f_valid["top"])  # calculamos la precision con el segmento de validacion
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy) / len(fold_accuracy)
    accuracies.append(avg)

# Mostramos los resultados obtenidos: nos dice para cada profundidad la precisión media. Nos quedamos con el que tenga la precisión más alta.
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
#print(df.to_string(index=False))
