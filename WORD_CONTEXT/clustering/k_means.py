# cluster word embeddings using K-means Since the Words are represented as vectors, applying KMeans is easy to do
# since the clustering algorithm will simply look at differences between vectors (and centers).
import argparse
import pathlib

import gensim.models
from sklearn.cluster import KMeans;
from sklearn.neighbors import KDTree;
import pandas as pd;
import numpy as np;
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt;
from itertools import cycle;


def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters, init='k-means++');
    idx = kmeans_clustering.fit_predict(word_vectors);
    print('classer ...')
    return kmeans_clustering, idx;


def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs);
    print(tree)
    # Closest points for each Cluster center is used to query the closest k points to it.
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers];
    # print(closest_points)
    closest_words_idxs = [x[1] for x in closest_points];
    # print("word index ")
    # print(closest_words_idxs)

    # Word Index is queried for each position in the above array, and added to a Dictionary.
    closest_words = {};
    for i in range(0, len(closest_words_idxs)):
        closest_words['Cluster #' + str(i)] = [index2word[j] for j in closest_words_idxs[i][0]]

    # A DataFrame is generated from the dictionary.
    df = pd.DataFrame(closest_words);
    df.index = df.index + 1
    return df;


def display_cloud(cluster_num, cmap, top_words, output_path):
    wc = WordCloud(background_color="white", max_words=2000, max_font_size=80, colormap=cmap, ranks_only=True);
    wordcloud = wc.generate(' '.join([word for word in top_words['Cluster #' + str(cluster_num)]]))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    cluster_name = 'cluster_' + str(cluster_num)
    plt.savefig(str(output_path / cluster_name), bbox_inches='tight')


def make_clouds(model, centers, num_clusters, num_top, file_name):
    output_path = file_name.parent / 'clusters'
    if not output_path.exists():
        output_path.mkdir(parents=True)
    top_words = get_top_words(model.index2word, num_top, centers, model.vectors);
    print('top_words ')
    print(top_words)
    cmaps = cycle([
        'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
        'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
        'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])

    for i in range(num_clusters):
        col = next(cmaps)
        display_cloud(i, col, top_words, output_path)


def kmeans_clustering(file_name, num_clusters):
    num_top = 50

    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
    Z = model.vectors

    print(Z[0].shape)

    kmeans_clustering, clusters = clustering_on_wordvecs(Z, num_clusters);
    centers = kmeans_clustering.cluster_centers_
    centroid_map = dict(zip(model.index2word, clusters));
    print('centers ')
    i = 0
    centerdict = dict()
    for center in centers:
        print("center " + str(i))
        words = model.similar_by_vector(center, topn=5)
        keys = []
        for k, v in words:
            keys.append(k)
        centerdict[i] = keys
        i += 1
    make_clouds(model, centers, num_clusters, num_top, file_name)
    return kmeans_clustering, model, centerdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--model_path", required=True, help="path of the file model")
    ap.add_argument("-m", "--max", required=True, help="max clusters")

    args = vars(ap.parse_args())
    file_path = pathlib.Path(args["model_path"])
    if not file_path.exists():
        assert False, 'File doesn\'t exist'
    print("Start building classifier ")
    kmeans_clustering(file_path, int(args["max"]))
    print("clouds built ")

# stopword = set(stopwords.words('french'))
# stopwords_path = 'stopwords.json';
# stop_list = json.loads(open(stopwords_path).read())
# top_frame = tfidf("../scripts/Corpus_2018.json", stopword=stopword, stop_list=stop_list)
# classes = doc_classification(top_frame, kmeans_clust, model)
# network(classes[:5], centers)
# h_clustering()
# kmeans_clust, model, centers = kmeans_clustering("article_2015", 15)
