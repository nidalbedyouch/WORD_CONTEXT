import pathlib

from sklearn.cluster import KMeans;
import matplotlib.pyplot as plt;
import gensim.models
import argparse


def elbow(x, max):
    print("Computing elbow model ...")
    Nc = range(1, max)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
    score
    plt.figure(200)
    plt.plot(Nc, score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.savefig("elbow.png")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("-p","--model_path",required=True,help="path of the file model")
    ap.add_argument("-m","--max",required=True,help="max clusters")

    args = vars(ap.parse_args())
    file_path = pathlib.Path(args["model_path"])
    if not file_path.exists():
        assert False, 'File doesn\'t exist'
    model = gensim.models.KeyedVectors.load_word2vec_format(str(file_path), binary=True)
    Z = model.vectors
    max = int(args["max"])

    elbow(Z,max)
