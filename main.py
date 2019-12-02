import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import jaccard_similarity_score


def create_sparse(data):
    mat = pd.DataFrame()
    for i, genres in enumerate(data):
        for genre in genres.split("|"):
            if genre in mat.columns:
                mat.at[i, genre] += 1
            else:
                mat[genre] = np.zeros(data.shape[0])
                mat.at[i, genre] += 1
    return mat


def jaccard_sim(df1, df2):
    temp1 = df1.ne(0.0) == 1.0
    temp2 = df2.ne(0.0) == 1.0
    intersect = (temp1 & temp2)
    intersection = intersect.sum().sum()
    union = (temp1.sum() + temp2.sum()).sum() - intersection
    return float(intersection)/float(union)


df = pd.read_csv("book_clean.csv")
train_data = df[:700]
test_data = df[700:]
print("No books: ", train_data.shape[0])

sparse_train = create_sparse(train_data["genres"])
gold_standard = create_sparse(test_data["genres"])

print("Jaccard test: ", jaccard_sim(sparse_train, sparse_train))

for col in sparse_train.columns:
    if col not in gold_standard.columns:
        gold_standard[col] = np.zeros(len(test_data))

for col in gold_standard.columns:
    if col not in sparse_train.columns:
        sparse_train[col] = np.zeros(len(train_data))
print("No genres in GS: ", gold_standard.shape)
print("No genres in Train: ", sparse_train.shape)

gold_standard.sort_index(axis=1, inplace=True) # = gold_standard.reindex(sorted(gold_standard.columns), axis=1)
sparse_train.sort_index(axis=1, inplace=True) # = sparse_train.reindex(sorted(sparse_train.columns), axis=1)

KNpipe = Pipeline([('vectorizer', CountVectorizer()), ('classifier', MultiOutputClassifier(OneVsRestClassifier(SVC(kernel='linear'))))])
KNpipe.fit(train_data["book_desc"], sparse_train)
prediction = KNpipe.predict(test_data["book_desc"])
pred_panda = pd.DataFrame(prediction, index=range(len(prediction)), columns=sparse_train.columns)
score = gold_standard.subtract(pred_panda).sum(axis=1)

pred_panda.sort_index(axis=1, inplace=True) # = pred_panda.reindex(sorted(pred_panda.columns), axis=1)
score = jaccard_sim(pred_panda, gold_standard)
print("Jaccard score: ", score)

'''
no_genres = defaultdict(int)
unique_genres = defaultdict(int)
for row in test_data["genres"]:
    genres = row.split("|")
    no_genres[len(genres)] += 1
    for genre in genres:
        unique_genres[genre] += 1

print("Amount of tags/book: ", no_genres)
plt.bar(range(len(no_genres)), list(no_genres.values()), align='center')
plt.xticks(range(len(no_genres)), list(no_genres.keys()))
print("Number of unique genres: ", len(unique_genres))
'''

'''
genres_predict = defaultdict(int)
unique_predict = defaultdict(int)

for row in prediction:
    genres = row.split("|")
    genres_predict[len(genres)] += 1
    for genre in genres:
        unique_predict[genre] += 1
no_predict = len(unique_predict)

print("Number of unique genres: ", no_predict)

plt.bar(range(len(genres_predict)), list(genres_predict.values()), align='center')
plt.xticks(range(len(genres_predict)), list(genres_predict.keys()))
'''
#plt.show()
