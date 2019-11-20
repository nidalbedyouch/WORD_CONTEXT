import argparse
import pathlib

import gensim
import unicodedata
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt;


def hcluster_on_wordvecs(model, method, metric):
    l = linkage(model.vectors, method=method, metric=metric)

    # calculate full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.ylabel('')
    plt.xlabel('word')

    dendrogram(
        l,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=16.,  # font size for the x axis labels
        leaf_label_func=lambda v: str(unicodedata.normalize('NFD', model.index2word[v]).encode('ascii', 'ignore'))
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-p","--model_path",required=True,help="path of the file model")
    ap.add_argument("-m","--method",required=False,help="linkage method : single | complete | average | centroid | "
                                                        "weighted |median | ward |")
    ap.add_argument("-e", "--metric", required=False, help="linkage metric example : euclidean,cosine")

    args = vars(ap.parse_args())
    file_path = pathlib.Path(args["model_path"])
    if not file_path.exists():
        assert False, 'File doesn\'t exist'

    model = gensim.models.KeyedVectors.load_word2vec_format(str(file_path) , binary=True)
    method = 'average'
    if args["method"] is not None:
        method = args["method"]
    metric = 'cosine'
    if args["metric"] is not None:
        metric = args["metric"]
    hcluster_on_wordvecs(model,method,metric)

