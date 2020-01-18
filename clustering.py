import spacy
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from helper import create_sparse
from sklearn.cluster import KMeans
from collections import defaultdict
nlp = spacy.load("en_core_web_md")


def most_similar(x):
    y = np.expand_dims(x, axis=0)
    most_sim = nlp.vocab.vectors.most_similar(y, n=10)
    best_keys = most_sim[0][0]
    res = []
    for key in best_keys:
        res.append(nlp.vocab[key])
    return res


def create_vectors(words):
    all_word_vectors = []
    all_words = []
    for word in words:
        # print(word.text)
        all_word_vectors.append(word.vector)
        all_words.append(word.text)
    return all_words, all_word_vectors


def plot_most_similar(all_word_vectors, all_words):
    words_embedded = TSNE(n_components=2).fit_transform(all_word_vectors)
    print(words_embedded.shape)
    fig, ax = plt.subplots()
    ax.scatter(words_embedded[:, 0], words_embedded[:, 1])
    for i, txt in enumerate(all_words):
        ax.annotate(txt, (words_embedded[i, 0], words_embedded[i, 1]))


def cluster(all_word_vectors, all_words, cluster_count):
    words_clustered = KMeans(n_clusters=cluster_count, random_state=None).fit(all_word_vectors)
    clusters = defaultdict(int)
    for i, word in enumerate(all_words):
        clusters[word] = words_clustered.labels_[i]
        # print(cluster_dict[word])
        # print(word + " ", words_clustered.labels_[i])
    return clusters


# df = pd.read_csv("book_clean_big.csv")
# gen_mat = create_sparse(df["genres"])
# words, vectors = create_vectors(nlp.vocab[w] for w in gen_mat.columns)
# sse = []
# k_max = 400
# print(len(vectors)/2)
# for k in range(1, k_max):
#     print("\rProgress: ", k, end="")
#     km = KMeans(n_clusters=k)
#     km.fit(vectors)
#     sse.append(km.inertia_)
#
# # Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list(range(1, k_max)), sse, '-o')
# plt.xlabel(r'Number of clusters *k*')
# plt.ylabel('Sum of squared distance')
# plt.show()

# df = pd.read_csv("book_clean.csv")
# gen_mat = create_sparse(df["genres"])
# freq_count = gen_mat.sum()
# print(most_similar(nlp.vocab[""]))
# words, vectors = create_vectors(nlp.vocab[w] for w in gen_mat.columns)
# cluster_dict = cluster(vectors, words)
# plot_most_similar(nlp.vocab[w] for w in gen_mat.columns)  # ["cheese", "goat", "sweden", "university", "computer"])
# plt.show()
