# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 10:16:36 2020

@author: Asvins
"""
import pandas as pd
datos=pd.read_csv('Grafo.csv',header = 0)

arr=[]
arr2=[]
lim_sup=0
lim_inf=0
mini = []
mini2=[]
PG=[]
MG=[]
distanciasR=[]

for j in range(4):
    fila=datos.iloc[j]
    for i in range(4):
        arr2.append(fila[i])
    arr.append(arr2)
    arr2=[]

print('Matriz rutas: \n',arr,'\n')

for i in range(4):
    for j in range(4):
        if arr[i][j] != 0:
            mini.append(arr[i][j])
    mini2.append(min(mini))
    mini=[]
print(mini2)
print('limite Inferior: ', sum(mini2))
minimoG=sum(mini2)

for j in range(4):
    
    PL=[]
    lim_supaux=0
    infaux=0    
    ML=[]    
    nodoM=9999
    listaM=[]    
    ifal=0
    ifals=[]
    yfals=[]
    k=0
    l=0
    
    if j != 0:
        lim_sup=j
        inf=0
        PL.append(lim_inf)
        PL.append(lim_sup)
        ML.append(arr[0][j])
        lim_supaux=lim_sup
        ifal=lim_inf
        ifals.append(ifal)
        yfals.append(lim_supaux)
        while k < 4:
            ifal = lim_supaux
            if ifal not in ifals :
                ifals.append(ifal)
                while l < 4:
                    if l not in yfals and l!=0:
                        if arr[ifal][l] != 0:
                            if arr[ifal][l] < nodoM:
                                nodoM = arr[ifal][l]
                                infaux=ifal
                                lim_supaux=l                        
                    l += 1
                    if l!=0 and lim_supaux not in yfals:
                        yfals.append(lim_supaux)
                        PL.append(lim_supaux)
                        ML.append(nodoM)
                        ifal=lim_supaux
            k += 1
            l=0
            nodoM=9999
            if len(ML)==3:
                ML.append(arr[lim_supaux][0])
    k=0
    MG.append(ML)
    PG.append(PL)
    distanciasR.append(sum(ML))
    
    
print("---------Datos procesados:--------\n")

print('Posiciones recorridas: ',PG)
print('Distancias recorridas: ',MG)
print('suma de Distancias recorridas: ',distanciasR)
    