from datetime import datetime
from ipykernel import kernel_protocol_version
import pandas as pd
from sklearn import cluster
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import csv
import gensim
import gensim.corpora as corpora

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


def obtener_datos():
    df = pd.read_csv('datos prep/test_p.csv')
    df = df[:300]
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

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


print(datetime.now())
f=obtener_datos()


l_ord=[]
global dict_clusters
dict_clusters = {}
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
print('Distancias iniciales calculadas')
print(datetime.now())
while len(l_ord)>1:
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
print(datetime.now())

df = pd.read_csv('datos prep/test_p.csv')
data = df.text.values.tolist()
data_words = list(sent_to_words(data))
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,
                             stop_words='english',             
                             lowercase=True,                   
                             token_pattern='[a-zA-Z0-9]{3,}') 
data_vectorized = vectorizer.fit_transform(data)
# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=10)
lda_output = lda_model.fit_transform(data_vectorized)

topicnames = ["Topic" + str(i) for i in range(lda_model.n_components)]
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic["dominant_topic"] = dominant_topic
print(df_document_topic)

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(lda_model.components_)
# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names_out()
df_topic_keywords.index = topicnames
# View
print(df_topic_keywords)