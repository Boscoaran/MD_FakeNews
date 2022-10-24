from datetime import datetime
from inflect import print3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

n_instancias = 20
global dict_clusters
k = 7

def calcular_distancias(l1, l2, l_ord):
    dist=np.subtract(l1, l2)
    dist=[round(abs(ele),2) for ele in dist]
    dnm=round(np.sum(dist), 2)
    return(l_ord, dnm)        

def add_ord(In, Im, Dnm, l_ord):
    a=0
    b=len(l_ord)
    while a < b:
        mid = (a+b)//2
        if Dnm < l_ord[mid][0]:
            b = mid
        else:
            a = mid+1
    l_ord.insert(a, [Dnm, In, Im])        
    return (l_ord)

def eliminar_distancias(i1, i2, l_ord):
    eliminar=[]
    for i in range (0, len(l_ord)):
        if l_ord[i][1]==i1 or l_ord[i][2]==i1 or l_ord[i][1]==i2 or l_ord[i][2]==i2:
            eliminar.append(i)       
    for i in sorted(eliminar, reverse=True): 
        del l_ord[i]
    return(l_ord)        

def calcular_cluster(l_i1, l_i2, i1, i2, nomb):
    nuevo_cluster=[]
    if 'c' in str(i1):
        clusters_i1 = dict_clusters[i1]
    else:
        clusters_i1 = [int(i1)]
    if 'c' in str(i2):
        clusters_i2 = dict_clusters[i2]
    else:
        clusters_i2 = [int(i2)]       
    nomb=('c-'+str(nomb))
    l_clusters=clusters_i1+clusters_i2
    dict_clusters.update({nomb: l_clusters})
    for i in range(0, len(l_i1)):
        nuevo_cluster.append((l_i1[i]+l_i2[i])/2)
    nuevo_cluster.insert(0, nomb)     
    return nuevo_cluster    


def preprocesar():
    df = pd.read_csv('datos prep/test_p.csv')
    df = df[:n_instancias]
    tfidfvectorizer = TfidfVectorizer(analyzer='word')
    tfidf_wm = tfidfvectorizer.fit_transform(df['text'])
    df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray())
    n=tfidf_wm.shape[0]
    l = [item for item in range(1, n+1)]
    df_tfidfvect.insert(0, "id", l)
    f = df_tfidfvect.values.tolist()
    return(f)

def find(element, matrix):
    id=[]
    for i in range(0, len(matrix)):
        if element==matrix[i][0]:
            id.append(i)
    return(id[0])        

def cohesion_interna(df_resultados, instancias):
    cohesion_particion=0
    cohesion_cluster=[]
    for idx in df_resultados.index:
        centroide=df_resultados['centroide'][idx]
        cohesion_cluster_acc=0
        for instacia in df_resultados['instancias'][idx]:
            pos_instacia=instancias[instacia-1][1:]
            cohesion_cluster_acc=cohesion_cluster_acc+distancia_cuadratica(centroide, pos_instacia)
        cohesion_cluster.append(cohesion_cluster_acc)
        cohesion_particion=cohesion_particion+cohesion_cluster_acc
    df_resultados['cohesion']=cohesion_cluster
    return(df_resultados, cohesion_particion)

def distancia_cuadratica(x, y):
    acc=0
    for a, b in zip(x, y):
        acc=acc+pow((a-b), 2)
    return acc    
              




#f=preprocesar()
f0=[[1,1,1,1,1,1],[2,1,1,1,1,2],[3,1,1,1,1,0],[4,10,10,10,10,10],[5,10,10,10,10,11],[6,10,10,10,10,9],[7,20,20,20,20,20],[8,20,20,20,20,21],[9,20,20,20,20,19],[10,15,15,15,15,15]] 
f=f0.copy()
l_ord=[]
dict_clusters={}
dnm_acc=0
l_dnm_med=[]
for n in range (0, len(f)):
    for m in range (n+1, len(f)):
        l1=np.array(f[n][1:])
        l2=np.array(f[m][1:])
        l_ord, dnm = calcular_distancias(l1, l2, l_ord)
        l_ord = add_ord(n+1, m+1, dnm, l_ord)
        dnm_acc=dnm_acc+dnm
dnm_med=dnm_acc/len(l_ord)
l_dnm_med.append(round(dnm_med,2))
nomb_cluster=1
while len(l_ord)>k:
    i1 = l_ord[0][1]
    i2 = l_ord[0][2]
    l_ord=eliminar_distancias(i1, i2, l_ord)
    id1=find(i1, f)
    id2=find(i2, f)
    nuevo_cluster=calcular_cluster(f[id1][1:], f[id2][1:], i1, i2, nomb_cluster)
    nomb_cluster=nomb_cluster+1
    instancias_eliminar=[id1, id2]
    for i in sorted(instancias_eliminar, reverse=True):
        del f[i]      
    for n in range (0, len(f)):
        l1=np.array(f[n][1:])
        l2=np.array(nuevo_cluster[1:])
        l_ord, dnm=calcular_distancias(l1, l2, l_ord)
        l_ord=add_ord(f[n][0], nuevo_cluster[0], dnm, l_ord)
        dnm_acc=dnm+dnm_acc 
    f.append(nuevo_cluster)
    dnm_med=dnm_acc/len(l_ord)
    l_dnm_med.append(round(dnm_med, 2)) 
np.savetxt("distancias.csv", l_dnm_med, delimiter=",")
with open('clusters.csv', 'w') as clusters_csv:
    writer = csv.writer(clusters_csv)
    for key in dict_clusters.keys():
        r = [key]+dict_clusters[key]
        writer.writerow(r)
cluster=[]
centroide=[]
instacias=[]
for i in f:
    if 'c' not in str(i[0]):
        c='i-'+str(i[0])
        dict_clusters.update({c: [i[0]]})
        i[0]=c
    cluster.append(i[0])
    centroide.append(i[1:])
    instacias.append(dict_clusters[i[0]])
    dic_resultados={'cluster': cluster, 'centroide': centroide, 'instancias': instacias}
    resultados=pd.DataFrame(data=dic_resultados, columns=['cluster', 'instancias', 'centroide'])
resultados, cohesion_particion=cohesion_interna(resultados, f0)
print(resultados)
print(cohesion_particion)