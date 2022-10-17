import math
import numpy as np
import matplotlib.pyplot as plt

def new_elems(minI, minJ):
    print("Entrando en new_elems")
    v1 = elems[minI]
    print(v1)
    v2 = elems[minJ]
    print(v2)
    n_vector = []
    for e in range(0, len(v1)):
        val = min(v1[e],v2[e])
        #val = (v1[e] + v2[e])/2
        n_vector.append(val)
    elems.append(tuple(n_vector))
    print("Saliendo de new_elems")

def initial_clusters():
    for i in range(0,len(elems)):
        clusters[elems[i]] = [i]

def create_cluster_union(minI, minJ, distance, itr):
    print("Entrando a cluster_union")
    itr_str = str(itr)
    instance = str(elems[len(elems)-1]) + itr_str
    strI = str(elems[minI]) + str(itr-1)
    if strI in clusters:
        aux = strI
    else:
        aux = elems[minI]
    a1 = clusters[aux]
    strJ = str(elems[minJ]) + str(itr-1)
    if strJ in clusters:
        aux1 = strJ
    else:
        aux1 = elems[minJ]
    a2 = clusters[aux1]
    cluster_group = a1 + a2
    print(cluster_group)
    clusters[instance] = cluster_group
    if (minI > minJ):
        elems.pop(minI)
        elems.pop(minJ)
    else:
        elems.pop(minJ)
        elems.pop(minI)
    print("El nuevo array de instancias es: " + str(elems))
    print("La lista de clusters es: " + str(clusters))
    print("Saliendo de cluster union")
    
def create_matrix():
    print("Entrando a create_matrix")
    matrix = []
    minimo = 9999
    for i in range(0,len(elems)):
        e1 = elems[i]
        row = []
        for j in range(0, len(elems)):
            if i == j:
                row.append(0)
            elif i < j:
                e2 = elems[j]
                value = 0
                for index in range(0, len(e1)):
                    value += (e1[index] + e2[index]) ** 2
                    fValue = round(math.sqrt(value),2)
                row.append(fValue)
                if minimo > fValue:
                    minimo = fValue
                    minI = i
                    minJ = j
            elif i > j:
                row.append(0)
        matrix.append(row)
    print("La matriz de distancias es " + str(matrix))
    print(minimo)
    print(minI, minJ)
    #plt.matshow(matrix)
    #plt.colorbar()
    #plt.show()
    print("Saliendo de create_matrix")
    return minI, minJ, minimo

elems = [(1,2,4,5,7),(3,6,2,2,9),(9,2,1,5,7),(2,0,1,2,5),(10,3,20,3,11)]
clusters = {}
initial_clusters()
print(clusters)
cluster_union = {}
k = 4

for iteration in range(0,k):
    i, j, d = create_matrix()
    new_elems(i, j)
    create_cluster_union(i, j, d, iteration)

print(clusters)










