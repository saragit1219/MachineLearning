# EJ 1 PYTHON Y PANDAS (Sara Barrera Romero)

import pandas as pd
import numpy

df = pd.read_csv("animal_center_vet.csv", sep=",")
print(df)

# 1.- ¿De qué raza ("breed") es el animal con identificador "A668644"?
df[df["animal_id"] == "A668644"]["breed"]
#--> Chihuahua Shorthair Mix

# 2.- ¿Cuáles el listado de perros de raza "Pit Bull Mix" y una edad mayor de 1 año (365 días)?
df[(df["breed"] == "Pit Bull Mix") & (df["age_upon_outcome"] > 365)]

# 3.- ¿Cuántos días ("age_upon_outcome") tiene el cachorro más pequeño de cada especie ("animal_type")?
df.groupby(["animal_type"])["age_upon_outcome"].min()
# --> Cat: 14 días, Dog: 2 días, Other: 28 días

# 4.- Repetir la anterior pregunta desglosado por categoría de edad (“age_category”) y obteniendo la media de días en vez del mínimo. Resolver con “pivot_table”
df.pivot_table(index="age_category", columns="animal_type", values="age_upon_outcome", aggfunc=numpy.mean)


# SEGUNDO FICHERO:
df_2 = pd.read_csv("animal_center_prop.csv", sep=",")
print(df_2)

# 1.- ¿Cuál es la media de edad ("age_upon_outcome") de cada sexo("sex_upon_outcome")?
df_merged = pd.merge(df, df_2, on="animal_id")
age_means = df_merged.groupby("sex_upon_outcome")["age_upon_outcome"].mean()
print(age_means)

# 2.- Escribir en un fichero "report.csv" con el top 10 de animales más pequeños (en edad, "age_upon_outcome") de color negro ("Black"). Pista para la ordenación: sort_values
df_ordered = df_merged.sort_values(by=['age_upon_outcome'])
df_black = df_ordered[(df_ordered["color"] == "Black")]
top10 = df_black.head(10)
top10.to_csv("report.csv", sep=",")


