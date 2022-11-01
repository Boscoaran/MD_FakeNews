from cmath import inf
import string
import time
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import gensim

n_instancias = 1000
n_topicos = 100
dict_clusters={}
l_ord=[]
k = 5
resultados=pd.DataFrame()
 
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
        n1=len(clusters_i1)
    else:
        clusters_i1 = [int(i1)]
        n1=1
    if 'c' in str(i2):
        clusters_i2 = dict_clusters[i2]
        n2=len(clusters_i2)
    else:
        clusters_i2 = [int(i2)]       
        n2=2
    nomb=('c-'+str(nomb))
    l_clusters=clusters_i1+clusters_i2
    dict_clusters.update({nomb: l_clusters})
    for i in range(0, len(l_i1)):
        nuevo_cluster.append(round((l_i1[i]*n1+l_i2[i]*n2)/(n1+n2),4))
    nuevo_cluster.insert(0, nomb)     
    return nuevo_cluster    


def cargar_datos():
    df = pd.read_csv('datos/test.csv')
    df = df[:n_instancias]
    df=df.drop(columns=['title', 'id', 'author'])
    df['text'], eliminar=preprocesar(df['text'])
    eliminar=list(dict.fromkeys(eliminar))
    df=df.drop(df.index[eliminar])
    return(df)

def preprocesar(c):
    wnl = WordNetLemmatizer()
    ps = PorterStemmer()
    remove=[]
    for i in range(len(c)):
        s = str(c[i])
        if not bool(s):
            remove.append(i)
        else:
            s = s.lower()
            s = s.translate(str.maketrans('','', string.punctuation))
            s = s.translate(str.maketrans('','', string.digits))
            s = s.split()
            stop_words=set(stopwords.words('english'))
            filtered_sentence = []
            for w in s:
                if w not in stop_words and len(w)>1 and w.isascii():
                    w = wnl.lemmatize(w)
                    w = ps.stem(w)
                    if len(w)>1:
                        filtered_sentence.append(w)
            if not bool(filtered_sentence):
                remove.append(i)  
            c[i]=filtered_sentence
    return(c, remove)                                

def lda():
    id2word = gensim.corpora.Dictionary(f0['text'])
    id2word.save("id2word")
    corpus = [id2word.doc2bow(doc) for doc in f0['text']]
    lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=n_topicos)
    doc_lda = lda_model[corpus]
    lda_model.save("lda_model")

    matrix = np.zeros((len(f0), n_topicos+1))
    i = 0
    for t in doc_lda:
        matrix[i][0] = int(i+1)
        for e in t:
            matrix[i][e[0]+1] = e[1]
        i+=1
    return(matrix)    

def text_to_topics(text):
    wnl = WordNetLemmatizer()
    ps = PorterStemmer()
    text = text.lower()
    text = text.translate(str.maketrans('','', string.punctuation))
    text = text.translate(str.maketrans('','', string.digits))
    s = text.split()
    stop_words=set(stopwords.words('english'))
    preprocesed_text = []
    for w in s:
        if w not in stop_words and len(w)>1 and w.isascii():
                w = wnl.lemmatize(w)
                w = ps.stem(w)
                if len(w)>1:
                    preprocesed_text.append(w)
    lda_model=gensim.models.LdaModel.load("lda_model")
    id2word=gensim.corpora.Dictionary.load("id2word")
    other_corpus = [id2word.doc2bow(preprocesed_text)]
    topics=lda_model[other_corpus[0]]
    topics_0=['nc']
    topics_0.extend(0 for i in range(n_topicos))
    for topic in topics:
        topics_0[topic[0]]=topic[1]  
    return(topics_0)        
            
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
            dnm = distancia_cuadratica(l1, l2)
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
        dnm=distancia_cuadratica(l1, l2)
        add_ord(f[n][0], nuevo_cluster[0], dnm)
    f.append(nuevo_cluster) 

def calcular_metricas():
    cluster=[]
    centroide=[]
    instacias=[]
    for i in f:
        if 'c' not in str(i[0]):
            c='i-'+str(int(i[0]))
            dict_clusters.update({c: [i[0]]})
            i[0]=c
        cluster.append(i[0])
        centroide.append(i[1:])
        instacias.append(dict_clusters[i[0]])
        dic_resultados={'cluster': cluster, 'centroide': centroide, 'instancias': instacias}
        global resultados
        resultados=pd.DataFrame(data=dic_resultados, columns=['cluster', 'instancias', 'centroide'])   

def imprimir_resultados():
    cohesion=cohesion_interna()
    disimilitud=disimilitud_externa()
    separabilidad=separabilidad_externa()
    print('\nResultados para '+str(k)+' clusters:\n')
    print(resultados)
    print('\n')
    print('Metricas de la particion:')
    print('     -Cohesion interna: ' + str(cohesion))
    print('     -Disimilitud externa: '+ str(disimilitud))
    print('     -Separabilidad: ' + str(separabilidad))

def add_new_input_cluster():
    print('\nIntroduce un texto en Inglés:\n')
    new_text=input()
    new_values=text_to_topics(new_text)
    cluster_cercano=inf
    for index, row in resultados.iterrows():
        d=distancia_cuadratica(row['centroide'], new_values[1:])
        if d < cluster_cercano:
            cluster_cercano = d
            cluster_id = index
    print('\nLa instancia se va a añadir al cluster:')        
    print(resultados.iloc[[cluster_id]])      
    print('\nEl nuevo centroide sería:')
    nuevo_centroide=[]
    num_clusters=len(resultados['instancias'][cluster_id])
    for i in range(0, len(new_values)-1):
        nuevo_centroide.append(abs(round((resultados['centroide'][cluster_id][i]*num_clusters+new_values[i+1])/(1+num_clusters),2)))
    print(nuevo_centroide)


#f0=[[1,1,1,1,1,1],[2,1,1,1,1,2],[3,1,1,1,1,0],[4,10,10,10,10,10],[5,10,10,10,10,11],[6,10,10,10,10,9],[7,20,20,20,20,20],[8,20,20,20,20,21],[9,20,20,20,20,19],[10,15,15,15,15,15]] 
start_time = time.time()
f0=cargar_datos()                              #obtiene los datos y los preprocesa para el algoritmo
print('Datos cargados y preprocesados')
f0=lda().tolist()
print('LDA completado')
f=f0.copy()                                     #f0 mantiene las instancias iniciales, f contiene las instancias/clusters a lo largo del algoritmo
distancias_iniciales()                          #calcula las disntancias iniciales ente instancias y las ordena [instancia1, instancia2, distancia]
print('Distancias iniciales calculadas')
for nuevo_cluster in range(len(f0)-k):          #iteraciones del algoritmo hasta que en que queden k clusters
    unir_clusters_cercanos(nuevo_cluster)       #une los dos clusters más cercanos, c-(nuevo_cluster) es el nombre del nuevo cluster (c-1, c-2...) si un cluster solo esta formado por una instancia sera i-numero instancia
calcular_metricas()                             #prepara los datos para calcular las metricas y diseña la matriz de resultados
imprimir_resultados()                           #imprime la matriz con los resultados y las metricas (cohesion interna, disimilitud externa y separabilidad externa)
print("\n--- %s seconds ---" % (time.time() - start_time))
add_new_input_cluster()
   