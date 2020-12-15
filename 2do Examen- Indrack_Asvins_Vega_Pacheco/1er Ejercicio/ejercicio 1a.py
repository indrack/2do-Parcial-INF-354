# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 08:57:07 2020

@author: Asvins
"""

from deap import creator
from deap import algorithms
from deap import tools
from deap import base

import array
import random
import numpy

import pandas as pd

grafo=pd.read_csv('Grafo.csv', sep=',' ,header = 0)

print(grafo.values)

fil=[]
col=[]
res = 4

for j in range(4):
    fila=grafo.iloc[j]
    for i in range(4):
        col.append(fila[i])
    fil.append(col)
    col=[]
print("matriz de rutas: ","\n", fil)
print("\n", "\n")

distM = fil
    
def evaluacion(individual):

    dist = distM[individual[-1]][individual[0]]
    for gene1, gene2 in zip(individual[0:-1], individual[1:]):
        dist += distM[gene1][gene2]
    return dist,

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(res), res)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=30)
toolbox.register("evaluate", evaluacion)

def main():
    random.seed(169)

    pop = toolbox.population(n=1000)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 100, stats=stats, 
                        halloffame=hof)
    
    return pop, stats, hof





if __name__ == "__main__":
    pop,stats,hof=main()
    print(hof)
    print(evaluacion(hof[0]))