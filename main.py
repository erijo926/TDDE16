import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Kolumnerna ar: book_authors, book_desc, book_edition,
# book_format, book_isbn, book_pages, book_rating,
# book_rating_count, book_review_count, book_title
# genres, image_url
df = pd.read_csv("book_data18.csv")
df["genres"] = df["genres"].fillna("unknown") # removes null genres
df["book_desc"] = df["book_desc"].fillna("No description") # removes null desc
df["genres"] = df["genres"].str.split("|")

# print(df["book_desc"].isnull().sum())
train_data = df[0:700]
test_data = df[701:]

gold_standard = [list(set(x)) for x in df["genres"]]
unique = []
for book in gold_standard:
    for genre in book:
        if genre not in unique:
            unique.append(genre)
#gen_arr = np.asarray(gold_standard)
#unique = np.unique(gen_arr)

pipe = Pipeline([('vectorizer', CountVectorizer()), ('naive_bayes', MultinomialNB())])

print(unique)
# party_counts1718 =  df["genres"].value_counts()
# party_counts1718.plot(kind='bar')
# plt.show()
