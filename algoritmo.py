import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import csv

n_instancias = 20
dict_clusters={}
l_ord=[]
k = 3
resultados=pd.DataFrame()

def calcular_distancias(l1, l2):
    dist=np.subtract(l1, l2)
    dist=[round(abs(ele),2) for ele in dist]
    dnm=round(np.sum(dist), 2)
    return(dnm)        

def add_ord(In, Im, Dnm):
    a=0
    b=len(l_ord)
    while a < b:
        mid = (a+b)//2
        if Dnm < l_ord[mid][0]:
            b = mid
        else:
            a = mid+1
    l_ord.insert(a, [Dnm, In, Im])        

def eliminar_distancias(i1, i2):
    eliminar=[]
    for i in range (0, len(l_ord)):
        if l_ord[i][1]==i1 or l_ord[i][2]==i1 or l_ord[i][1]==i2 or l_ord[i][2]==i2:
            eliminar.append(i)       
    for i in sorted(eliminar, reverse=True): 
        del l_ord[i]       

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
    datos = df_tfidfvect.values.tolist()
    return(datos)

def find(element, matrix):
    id=[]
    for i in range(0, len(matrix)):
        if element==matrix[i][0]:
            id.append(i)
    return(id[0])        

def cohesion_interna():
    cohesion_particion=0
    cohesion_cluster=[]
    for idx in resultados.index:
        centroide=resultados['centroide'][idx]
        cohesion_cluster_acc=0
        for instacia in resultados['instancias'][idx]:
            pos_instacia=f0[int(instacia-1)][1:]
            cohesion_cluster_acc=cohesion_cluster_acc+distancia_cuadratica(centroide, pos_instacia)
        cohesion_cluster.append(cohesion_cluster_acc)
        cohesion_particion=cohesion_particion+cohesion_cluster_acc
    resultados['cohesion interna']=cohesion_cluster
    return(cohesion_particion)

def disimilitud_externa():
    dep=0           #disimilitud externa particion
    l_dec=[]        #lista disimilitud externa cluster
    for idx in resultados.index:
        centroide=resultados['centroide'].drop(idx)
        dec=0       #disimilitud externa cluster  
        for instancia in resultados['instancias'][idx]:
            pos_instancia=f0[int(instancia-1)][1:]
            dei=0       #disimilitud externa instancia
            for c in centroide:
                dei=dei+distancia_cuadratica(c, pos_instancia)
            dec=dec+dei        
        dep=dep+dec    
        l_dec.append(dec)
    resultados['disimilitud externa']=l_dec
    return(dep)          

def separabilidad_externa():
    centroide_conjunto=[]
    l_centroides=resultados['centroide']
    for i in range(0, len(l_centroides[0])):
        acc_centroide=0
        for centroide in l_centroides:
            acc_centroide=acc_centroide+centroide[i]
        centroide_conjunto.append(acc_centroide/len(l_centroides))
    separabilidad_particion=0    
    for centroide in l_centroides:
        separabilidad_particion=distancia_cuadratica(centroide_conjunto, centroide)+separabilidad_particion
    separabilidad_particion=separabilidad_particion*len(l_centroides)    
    return(separabilidad_particion)  

def distancia_cuadratica(x, y):
    acc=0
    for a, b in zip(x, y):
        acc=acc+pow((a-b), 2)
    return acc    
              
def distancias_iniciales():
    for n in range (0, len(f)):
        for m in range (n+1, len(f)):
            l1=np.array(f[n][1:])
            l2=np.array(f[m][1:])
            dnm = calcular_distancias(l1, l2)
            add_ord(n+1, m+1, dnm)

def unir_clusters_cercanos(n_cluster):
    i1 = l_ord[0][1]
    i2 = l_ord[0][2]
    eliminar_distancias(i1, i2)
    id1=find(i1, f)
    id2=find(i2, f)
    nuevo_cluster=calcular_cluster(f[id1][1:], f[id2][1:], i1, i2, n_cluster)
    instancias_eliminar=[id1, id2]
    for i in sorted(instancias_eliminar, reverse=True):
        del f[i] 
    for n in range (0, len(f)):
        l1=np.array(f[n][1:])
        l2=np.array(nuevo_cluster[1:])
        dnm=calcular_distancias(l1, l2)
        add_ord(f[n][0], nuevo_cluster[0], dnm)
    f.append(nuevo_cluster) 

def calcular_metricas():
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
        global resultados
        resultados=pd.DataFrame(data=dic_resultados, columns=['cluster', 'instancias', 'centroide'])   

def imprimir_resultados():
    print('Resultados para '+str(k)+' clusters:')
    print(resultados)
    print('\n')
    print('Metricas de la particion:')
    print('     -Cohesion interna: ' + str(cohesion_interna()))
    print('     -Sisimilitud externa: '+ str(disimilitud_externa()))
    print('     -Separabilidad: ' + str(separabilidad_externa()))

#f0=[[1,1,1,1,1,1],[2,1,1,1,1,2],[3,1,1,1,1,0],[4,10,10,10,10,10],[5,10,10,10,10,11],[6,10,10,10,10,9],[7,20,20,20,20,20],[8,20,20,20,20,21],[9,20,20,20,20,19],[10,15,15,15,15,15]] 
f0=preprocesar()                                #obtiene los datos y los preprocesa para el algoritmo
f=f0.copy()                                     #f0 mantiene las instancias iniciales, f contiene las instancias/clusters a lo largo del algoritmo
distancias_iniciales()                          #calcula las disntancias iniciales ente instancias y las ordena [instancia1, instancia2, distancia]
for nuevo_cluster in range(len(f0)-k):          #iteraciones del algoritmo hasta que en que queden k clusters
    unir_clusters_cercanos(nuevo_cluster)       #une los dos clusters más cercanos, c-(nuevo_cluster) es el nombre del nuevo cluster (c-1, c-2...) si un cluster solo esta formado por una instancia sera i-numero instancia
calcular_metricas()                             #prepara los datos para calcular las metricas y diseña la matriz de resultados
imprimir_resultados()                           #imprime la matriz con los resultados y las metricas (cohesion interna, disimilitud externa y separabilidad externa)
