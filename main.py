import pandas as pd
df = pd.read_csv("book_data18.csv")
# Kolumnerna är: book_authors, book_desc, book_edition,
# book_format, book_isbn, book_pages, book_rating,
# book_rating_count, book_review_count, book_title
# genres, image_url

print(df["book_title"])
