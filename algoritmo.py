import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

def calcular_cluster(l_i1, l_i2, i1, i2):
    nuevo_cluster=[]
    for i in range(0, len(l_i1)):
        nuevo_cluster.append((l_i1[i]+l_i2[i])/2)
    nuevo_cluster.insert(0, str(str(i1)+"_"+str(i2)))     
    return nuevo_cluster    


def obtener_datos():
    df = pd.read_csv('datos prep/test_p.csv')
    df = df[:100]
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


f=obtener_datos()
l_ord=[]
dnm_acc=0
for n in range (0, len(f)):
    for m in range (n+1, len(f)):
        l1=np.array(f[n][1:])
        l2=np.array(f[m][1:])
        l_ord, dnm = calcular_distancias(l1, l2, l_ord)
        l_ord = add_ord(n, m, dnm, l_ord)
        dnm_acc=dnm_acc+dnm
dnm_med=dnm_acc/len(l_ord)
while len(f)>1:
    i1 = l_ord[0][1]
    i2 = l_ord[0][2]
    l_ord=eliminar_distancias(i1, i2, l_ord)
    id1=find(i1, f)
    id2=find(i2, f)
    nuevo_cluster=calcular_cluster(f[id1][1:], f[id2][1:], i1, i2)
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
    print(nuevo_cluster[0])