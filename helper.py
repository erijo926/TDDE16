import pandas as pd
import numpy as np

def create_sparse(data):
    mat = pd.DataFrame()
    for i, genres in enumerate(data):
        for genre in genres.split("|"):
            if genre in mat.columns:
                mat.at[i, genre] = 1
            else:
                mat[genre] = np.zeros(data.shape[0])
                mat.at[i, genre] = 1
    return mat


def create_sparse_cluster(data, cluster_dict):
    mat = pd.DataFrame()
    for i, genres in enumerate(data):
        for genre in genres.split("|"):
            cluster = cluster_dict[genre]
            # print(genre, ": ", cluster)
            if cluster in mat.columns:
                mat.at[i, cluster] = 1
            else:
                mat[cluster] = np.zeros(data.shape[0])
                mat.at[i, cluster] = 1
    return mat


def create_naive(data):
    mat = pd.DataFrame()
    for i, genres in enumerate(data):
        for genre in genres.split("|"):
            if len(genre.split()) > 1:
                break
            else:
                if genre in mat.columns:
                    mat.at[i, genre] += 1
                else:
                    mat[genre] = np.zeros(data.shape[0])
                    mat.at[i, genre] += 1
    return mat


def jaccard_sim(df1, df2):
    temp1 = df1.ne(0.0) == 1.0
    temp2 = df2.ne(0.0) == 1.0
    intersect = (temp1 & temp2)
    intersection = intersect.sum().sum()
    union = (temp1.sum() + temp2.sum()).sum() - intersection
    return intersection/float(union)


def accuracy(df1, df2):
    numer = (df1 == df2).sum().sum()
    denom = df1.size
    return numer/float(denom)
