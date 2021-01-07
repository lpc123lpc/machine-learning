import pandas as pd
import gensim
import random
# numpy
import numpy as np

# classifier
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import csv

from sklearn.preprocessing import StandardScaler

def pre_treatment(reviews):
    a = []
    for review in reviews:
        review = review.lower()
        review = review.replace('\n', '')
        review = review.replace('<br />', ' ')
        #   用空格隔开单词和标点
        puncs = """()[]{}.,:;*?!"""
        for punc in puncs:
            review = review.replace(punc, ' ' + punc + ' ')
        review = review.split()
        a.append(review)

    return a


def lable_reviews(reviews, labels):
    a = []
    for i, review in enumerate(reviews):
        a.append(TaggedDocument(review, [labels[i]]))
    return a


def combine(dm, dbow, reviews, labels):
    dm_vec = np.zeros((len(reviews), 100))
    dbow_vec = np.zeros((len(reviews), 100))
    for i in range(len(reviews)):
        dm_vec[i] = dm.docvecs[labels[i]]
        dbow_vec[i] = dbow.docvecs[labels[i]]
    vecs = np.hstack((dm_vec, dbow_vec))

    return vecs


df = pd.read_csv('C:\\Users\\lpc\\Desktop\\public_data\\train.csv',index_col=0)
df2 = pd.read_csv('C:\\Users\\lpc\\Desktop\\public_data\\test_data.csv',index_col=0)

train_tag = df.index
X_train = df['review']
y_train = df['sentiment']

test_tag = df2.index
X_test = df2['review']

X_train = pre_treatment(X_train)
X_test = pre_treatment(X_test)

x_train_labels = list(lable_reviews(X_train,train_tag))
x_test_labels = list(lable_reviews(X_test,test_tag))

all_tag = x_train_labels
all_tag.extend(x_test_labels)

size = 100
model_dm = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size,
                                 sample=1e-3, negative=5, workers=3, epochs=10)
model_dbow = gensim.models.Doc2Vec(min_count=1, window=10, vector_size=size,
                                   sample=1e-3, negative=5, dm=0, workers=3, epochs=10)
# 对所有评论创建词汇表

model_dm.build_vocab(all_tag)
model_dbow.build_vocab(all_tag)


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return shuffled


for epoch in range(10):
    model_dm.train(sentences_perm(all_tag), total_examples=model_dm.corpus_count, epochs=1)
    model_dbow.train(sentences_perm(all_tag), total_examples=model_dbow.corpus_count, epochs=1)

x_train_vec = combine(model_dm,model_dbow,X_train,train_tag)
x_test_vec = combine(model_dm,model_dbow,X_test,test_tag)

regression = LogisticRegression()
regression.fit(x_train_vec, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
y_pred = regression.predict_proba(x_test_vec)

result = pd.read_csv('C:\\Users\\lpc\\Desktop\\public_data\\submission.csv')
result.loc[:,'sentiment'] = regression.predict(x_test_vec)
result.to_csv('C:\\Users\\lpc\\Desktop\\public_data\\submission2.csv')


