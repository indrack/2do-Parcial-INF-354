# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:48:27 2020

@author: Asvins
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


# Carga del dataset
data = pd.read_csv("crx.data", header=None)

# Viendo los data 
print(data)
print("\n")

# Estadisticas de los data numericos
print('-------------Estadisticas------------')
data_descr = data.describe()
print(data_descr)

print("\n")

# viendo los valores Nulos o Faltantes

import numpy as np

print('-------------Numero de NUlls------------')
# Comprobacion y Sumatoria de la Data nula
print(data.isnull().values.sum())

# Remplazamos los valores '?' con NaN
data = data.replace("?",np.NaN)

# Imputamos los valores NAN con la media
data = data.fillna(data.mean())

print('-------------Numero de NUlls 2------------')
# Contamos le numero de NUll de nuevo para verificar
print(data.isnull().values.sum())

# En todas las columnas e imputamos los valores NAN 

for col in data.columns:
    #vemos si es de tipo object - nominal
    if data[col].dtypes == 'object':
        # imputamos con la moda
        data[col] = data[col].fillna(data[col].value_counts().index[0])

# Contamos otra vez los valores NAN 

print('-------------Numero de NUlls 3------------')
print(data.isnull().values.sum())

print("\n")
print('-------------Preprocesamiento 1------------')

#Categorizamos
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# con todas las tablas cambiamos las valores no numericos a numericos en un rango 
for col in data.columns:
    #vemos si es de tipo object - nominal
    if data[col].dtype=='object':
    # tranfomramos los valores no numericos
        data[col]=le.fit_transform(data[col])
        
print(data)      
       

print("\n")
print('-------------Preprocesamiento 2------------')
# Normalizamos con  la libreria (minmaxscaler)

from sklearn.preprocessing import MinMaxScaler

# Elimina los campos 10 y 13 y convierta el DataFrame en una matriz NumPy
data = data.drop([data.columns[10],data.columns[13]], axis=1)

print(data)

data = data.values

# Separa los campos y etiquetas en variables separadas
X,y = data[:,0:13], data[:,13]

#  instanciamos  MinMaxScaler y reescalonamos
esc = MinMaxScaler(feature_range=(0,1))
rescX = esc.fit_transform(X)

print('-------------Seleccion Entrenamiento 80% y Test 20%------------')
# Entrenaminto y Test de la Red Neuronal
from sklearn.model_selection import train_test_split

# Almacenamos nuestros Train´s y TEST´s en 2 partes
Xtrain, Xtest, ytrain, ytest = train_test_split(rescX,  y,   test_size=0.20,     random_state=42)
print('---------Tamaños----------')
print("Train: ", len(Xtrain))
print("Test: ", len(Xtest))

print('----------- Red Neuronal Clasificador Multilayer Perceptron------------')
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=300)
clf.fit(Xtrain, ytrain)
Ypred = clf.predict(Xtest)

print('--------TEST--------')
print(ytest)
print('-------Prediccion------')
print(Ypred)
print()
print('-------------Matriz de confusion-----------')
print(confusion_matrix(ytest, Ypred))
print("Exactitud: ", accuracy_score(ytest, Ypred)-100)

logreg = LogisticRegression()
logreg.fit(Xtrain,ytrain)
y_pred = logreg.predict(Xtest)
print("----------- Precisión del clasificador de regresión logística:----------- ", logreg.score(Xtest, ytest))
print(confusion_matrix(ytest, y_pred), "\n")


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
print("----------- Ejemplos usados para entrenar TRAIN y TEST -X-: -----------", "\n")
print('Xtrain: ',len(Xtrain), "\n")
print(Xtrain)
print('Xtest: ',len(Xtest), "\n")
print(Xtest)
print("----------- Ejemplos usados para entrenar TRAIN y TEST -Y-: -----------", "\n")
print('ytrain: ',len(ytrain), "\n")
print(ytrain)
print(' ytest: ',len(ytest), "\n")
print(ytest, "\n")

classifier = MLPClassifier(activation="relu",solver='adam', hidden_layer_sizes=(100,),max_iter=100)
classifier.fit(Xtrain, ytrain)
y_pred = classifier.predict(Xtest)
print('\n Y_PREDICC: ',y_pred, "\n")

accuracy = confusion_matrix(ytest, Ypred)
accuracies = cross_val_score(estimator = classifier, X = Xtrain, y = ytrain, cv = 10)
print("Datos estadísticos de las precisiones\n")
print('Media de las precisiones después de la validación cruzada: ', accuracies.mean(), "\n")
print('Desviación estándar dentro de las precisiones: ', accuracies.std(), "\n")
print('precisiones: ', accuracies)
