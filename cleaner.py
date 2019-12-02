import pandas as pd
from langdetect import detect

# Kolumnerna ar: book_authors, book_desc, book_edition,
# book_format, book_isbn, book_pages, book_rating,
# book_rating_count, book_review_count, book_title
# genres, image_url
df = pd.read_csv("book_data.csv")
print("Size of dataset: ", df.shape[0])

df = df[pd.notnull(df["book_desc"])]
#df.dropna(subset=["book_desc"])
print("Null desc removed: ", df.shape[0])
df = df[pd.notnull(df["genres"])]
#df.dropna(subset=["genres"])
print("Null genres removed: ", df.shape[0])

invalid_desc = []
for i in df.index:
    desc = unicode(df.at[i, "book_desc"], "utf-8")
    #print(desc)
    try:
        if detect(desc) != "en":
            invalid_desc.append(i)
    except:
        invalid_desc.append(i)

df = df.drop(index=invalid_desc)
print("Non-english works removed: ", df.shape[0])

df.to_csv(path_or_buf="book_clean_big.csv")
