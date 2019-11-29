import pandas as pd
from langdetect import detect

# Kolumnerna ar: book_authors, book_desc, book_edition,
# book_format, book_isbn, book_pages, book_rating,
# book_rating_count, book_review_count, book_title
# genres, image_url
df = pd.read_csv("book_data18.csv")
print("Size of dataset: ", df.shape[0])

df = df[pd.notnull(df["book_desc"])]
#df.dropna(subset=["book_desc"])
print("Null desc removed: ", df.shape[0])
df = df[pd.notnull(df["genres"])]
#df.dropna(subset=["genres"])
print("Null genres removed: ", df.shape[0])

for i,text in enumerate(df["book_desc"]):
    desc = unicode(text, "utf-8")
    if detect(desc) != "en":
        print(desc)
        df = df.drop(index=i)
print("Non-english works removed: ", df.shape[0])

df.to_csv(path_or_buf="book_clean.csv")
