from sre_compile import isstring
import pandas as pd
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop = stopwords.words('english')

df = pd.read_csv('test.csv')
