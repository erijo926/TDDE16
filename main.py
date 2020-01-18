import pandas as pd
import numpy as np
import spacy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from helper import create_sparse, jaccard_sim, accuracy, create_naive, create_sparse_cluster
from clustering import cluster, create_vectors

from sklearn.svm import SVC
from collections import defaultdict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

nlp = spacy.load("en_core_web_md")

# df = pd.read_csv("book_clean.csv")
# train_data = df[:700]
# test_data = df[700:]
train_data = pd.read_csv("book_clean_big.csv")
test_data = pd.read_csv("book_clean.csv")

print("No books: ", train_data.shape[0])

sparse_train = create_sparse(train_data["genres"])
gold_standard = create_sparse(test_data["genres"])
# sparse_train = create_naive(train_data["genres"])
# gold_standard = create_naive(test_data["genres"])

temp = sparse_train.sum(axis=0)
temp_sort = temp.sort_values(ascending=False)

for col in sparse_train.columns:
    if col not in gold_standard.columns:
        gold_standard[col] = np.zeros(len(test_data))

for col in gold_standard.columns:
    if col not in sparse_train.columns:
        sparse_train[col] = np.zeros(len(train_data))

# assert gold_standard.shape[1] == sparse_train.shape[1]
print("No genres in GS: ", gold_standard.shape[1])
print("No genres in Train: ", sparse_train.shape[1])

words, vectors = create_vectors(nlp.vocab[w] for w in sparse_train.columns)
cluster_dict = cluster(vectors, words, 200)  # Clusters the genres according to spacys vocab vectors.

clustered_train = create_sparse_cluster(train_data["genres"], cluster_dict)
clustered_gold = create_sparse_cluster(test_data["genres"], cluster_dict)

print("No clusters in GS: ", clustered_train.shape)
print("No clusters in Train: ", clustered_gold.shape)

gold_standard.sort_index(axis=1, inplace=True)
sparse_train.sort_index(axis=1, inplace=True)
clustered_train.sort_index(axis=1, inplace=True)
clustered_gold.sort_index(axis=1, inplace=True)

# KNpipe = Pipeline([('vectorizer', CountVectorizer(stop_words="english")),
#                    ('classifier', MultiOutputClassifier(LogisticRegression(solver="sag", multi_class="ovr")))])
KNpipe = Pipeline([('vectorizer', CountVectorizer(stop_words="english")),
                   ('classifier', MultiOutputClassifier(MultinomialNB()))])

# ------------ With clustering
KNpipe.fit(train_data["book_desc"], clustered_train)

prediction = KNpipe.predict(test_data["book_desc"])
pred_cluster = pd.DataFrame(prediction, index=range(len(prediction)), columns=clustered_train.columns)

pred_cluster.sort_index(axis=1, inplace=True)  # = pred_panda.reindex(sorted(pred_panda.columns), axis=1)
print("Jaccard sim, clustered: %.4f" % jaccard_sim(pred_cluster, clustered_gold))
# print("Accuracy, clustered: %.4f" % accuracy(pred_cluster, clustered_gold))


# ------------ Without clustering
# KNpipe.fit(train_data["book_desc"], sparse_train)
#
# prediction = KNpipe.predict(test_data["book_desc"])
# pred_panda = pd.DataFrame(prediction, index=range(len(prediction)), columns=sparse_train.columns)
# score = gold_standard.subtract(pred_panda).sum(axis=1)
#
# pred_panda.sort_index(axis=1, inplace=True)  # = pred_panda.reindex(sorted(pred_panda.columns), axis=1)
# print("Jaccard sim: %.4f" % jaccard_sim(pred_panda, gold_standard))
# print("Accuracy: %.4f" % accuracy(pred_panda, gold_standard))