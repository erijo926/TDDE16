import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from helper import create_sparse, jaccard_sim, accuracy


df = pd.read_csv("book_clean.csv")
train_data = df[:700]
test_data = df[700:]
print("No books: ", train_data.shape[0])

sparse_train = create_sparse(train_data["genres"])
gold_standard = create_sparse(test_data["genres"])

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

gold_standard.sort_index(axis=1, inplace=True)
sparse_train.sort_index(axis=1, inplace=True)

# KNpipe = Pipeline([('vectorizer', CountVectorizer(stop_words="english")),
#                    ('classifier', MultiOutputClassifier(OneVsRestClassifier(LogisticRegression(solver="sag"))))])
KNpipe = Pipeline([('vectorizer', CountVectorizer(stop_words="english")),
                   ('classifier', MultiOutputClassifier(MultinomialNB()))])
KNpipe.fit(train_data["book_desc"], sparse_train)

prediction = KNpipe.predict(test_data["book_desc"])
pred_panda = pd.DataFrame(prediction, index=range(len(prediction)), columns=sparse_train.columns)
score = gold_standard.subtract(pred_panda).sum(axis=1)

pred_panda.sort_index(axis=1, inplace=True)  # = pred_panda.reindex(sorted(pred_panda.columns), axis=1)
print("Jaccard sim: %.4f" % jaccard_sim(pred_panda, gold_standard))
print("Accuracy: %.4f" % accuracy(pred_panda, gold_standard))