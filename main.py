import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer


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


df = pd.read_csv("book_clean.csv")
print("No of books: ", df.shape[0])
train_data = df[0:700]
test_data = df[701:]

sparse_genre = create_sparse(train_data["genres"])

no_genres = defaultdict(int)
unique_genres = defaultdict(int)
gold_standard = test_data["genres"]
for row in gold_standard:
    genres = row.split("|")
    no_genres[len(genres)]+=1
    for genre in genres:
        unique_genres[genre]+=1

print("Amount of tags/book: ", no_genres)
plt.bar(range(len(no_genres)), list(no_genres.values()), align='center')
plt.xticks(range(len(no_genres)), list(no_genres.keys()))
print("Number of unique genres: ", len(unique_genres))

KNpipe = Pipeline([('vectorizer', CountVectorizer()), ('classifier', KNeighborsClassifier())])
KNpipe.fit(train_data["book_desc"], train_data["genres"]) #.astype("str")
prediction = KNpipe.predict(test_data["book_desc"])
score = KNpipe.score(test_data["book_desc"], test_data["genres"])

genres_predict = defaultdict(int)
unique_predict = defaultdict(int)
for row in prediction:
    genres = row.split("|")
    genres_predict[len(genres)]+=1
    for genre in genres:
        unique_predict[genre]+=1

no_predict = len(unique_predict)
print("Number of unique genres: ", no_predict)
print(score)
plt.bar(range(len(genres_predict)), list(genres_predict.values()), align='center')
plt.xticks(range(len(genres_predict)), list(genres_predict.keys()))
#plt.show()

#print(classification_report(gold_standard, prediction)) #,target_names=target_names
"""
# party_counts1718 =  df["genres"].value_counts()
# party_counts1718.plot(kind='bar')
# plt.show()
"""