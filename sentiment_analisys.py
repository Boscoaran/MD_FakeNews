from itertools import count
import pandas as pd
import string
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

def normalize(d):

    lemmatazier = WordNetLemmatizer()
    ps = PorterStemmer()
    for n in range(len(d)):
        s = str(d[n])
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = s.split()
        stop_words=set(stopwords.words('english'))
        filtered_sentence = [w for w in s if not w in stop_words]
        filtered_sentence = []
        for w in s:
            if w not in stop_words and len(w)>1:
                w = lemmatazier.lemmatize(w)
                w = ps.stem(w)
                filtered_sentence.append(w)
        s = filtered_sentence
        d[n]=s
    return (d)
  
def preprocess(dataset):
    dataset['text']=normalize(dataset['text'])
    dataset['title']=normalize(dataset['title']) 
    return(dataset)  

def contar(list):
    l = []
    for i in list:
        l.append(len(i))
    return l    

if __name__=='__main__':
    df=pd.read_csv('test.csv')
    df=preprocess(df)
    df['len'] = contar(df['text'])
    #print(df['len'])
    print(df['len'].mean())
  