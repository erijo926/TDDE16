import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# Kolumnerna ar: book_authors, book_desc, book_edition,
# book_format, book_isbn, book_pages, book_rating,
# book_rating_count, book_review_count, book_title
# genres, image_url
df = pd.read_csv("book_data18.csv")
df["genres"] = df["genres"].fillna("Not Found")  # removes null genres

df["book_desc"] = df["book_desc"].fillna("No description")  # removes null desc

# print(df["book_desc"].isnull().sum())
train_data = df[0:700]
test_data = df[701:]

temp_genre = df["genres"].str.split("|")
gold_standard = [list(set(x)) for x in temp_genre]
unique_genres = []
for book in gold_standard:
    for genre in book:
        if genre not in unique_genres:
            unique_genres.append(genre)

gold_standard = test_data["genres"]
pipe = Pipeline([('vectorizer', CountVectorizer()), ('classifier', KNeighborsClassifier())])
pipe.fit(train_data["book_desc"], train_data["genres"]) #.astype("str")
prediction = pipe.predict(test_data["book_desc"])
score = pipe.score(test_data["book_desc"], gold_standard)
print(score)
mlb = MultiLabelBinarizer()
true = mlb.fit_transform(gold_standard[701:])
#target_names = unique_genres
print(classification_report(gold_standard, prediction)) #,target_names=target_names

# party_counts1718 =  df["genres"].value_counts()
# party_counts1718.plot(kind='bar')
# plt.show()
