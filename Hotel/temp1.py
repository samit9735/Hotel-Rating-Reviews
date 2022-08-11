import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words("English"))
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pickle
df = pd.read_csv('D:\\project\\hotels\\hotel_reviews.csv',encoding="ISO-8859-1")
df['Review'] = df['Review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

df['Review'] = df['Review'].str.split()
df['Review'] = df['Review'].apply(lambda x: [stemmer.stem(y) for y in x])


def make_sentences(data, name):
    data[name] = data[name].apply(lambda x: ' '.join([i + ' ' for i in x]))
    data[name] = data[name].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))


make_sentences(df, 'Review')


def sentiment(review):
    if review >= 3:
        return 1
    else:
        return 0


df['sentiment'] = df['Rating'].apply(sentiment)
X = df['Review']
Y = df['sentiment']

tfIdfVectorizer = TfidfVectorizer(use_idf=True)
X = tfIdfVectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X, Y)
pickle.dump(model, open('D:\\project\\hotels\\reviews.sav', 'wb'))
pickle.dump(tfIdfVectorizer, open('D:\\project\\hotels\\/rating.sav', 'wb'))
results = pickle.load(open('D:\\project\\hotels\\reviews.sav', 'rb')).score(X, Y)
print(results)