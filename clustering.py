import spacy
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from helper import create_sparse
nlp = spacy.load("en_core_web_md")


def most_similar(x):
    y = np.expand_dims(x, axis=0)
    most_sim = nlp.vocab.vectors.most_similar(y, n=10)
    best_keys = most_sim[0][0]
    res = []
    for key in best_keys:
        res.append(nlp.vocab[key])
    return res


def plot_most_similar(words):
    all_word_vectors = []
    all_words = []
    for word in words:
        print(word.text)
        all_word_vectors.append(word.vector)
        all_words.append(word.text)
        most_similar_words = most_similar(word.vector)
        for similar_word in most_similar_words:
            all_word_vectors.append(similar_word.vector)
            all_words.append(similar_word.text)

    words_embedded = TSNE(n_components=2).fit_transform(all_word_vectors)
    # print(words_embedded.shape)
    fig, ax = plt.subplots()
    ax.scatter(words_embedded[:, 0], words_embedded[:, 1])
    for i, txt in enumerate(all_words):
        ax.annotate(txt, (words_embedded[i, 0], words_embedded[i, 1]))


df = pd.read_csv("book_clean.csv")
gen_mat = create_sparse(df["genres"])
plot_most_similar(nlp.vocab[w] for w in gen_mat.columns)  # ["cheese", "goat", "sweden", "university", "computer"])
plt.show()
