# A script to do t-sne with word vector data
# Using sklearn t-sne


import csv
import json
import sys
from pathlib import Path
import getopt

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne_reduction(source_path):
    output_path = source_path.parent / Path("vector_result.json")
    output_path_csv = source_path.parent / Path("vector_result.csv")
    output_img = source_path.parent / Path("tsne.png")

    # Load raw json data
    raw_vectors = json.loads(open(str(source_path)).read())

    # Create two list to store words and their vectors separately
    vector_list = list()
    word_list = list()
    for value in raw_vectors.values():
        vector_list.append(value)
    for key in raw_vectors.keys():
        word_list.append(key)

    # TSNE part
    # Create a numpy array from vector list()
    X = np.asarray(vector_list).astype('float64')
    # Convert it to a 3 dimensional vector space
    # Parameters matters
    tsne_model = TSNE(perplexity=15, n_components=2, verbose=1, init='pca', n_iter=500, method='exact')
    np.set_printoptions(suppress=True)
    # .fit_transform: fit X into an embedded space and return that transformed output
    # .tolist(): use tolist() to convert numpy array into python list data structure
    y = tsne_model.fit_transform(X)
    sizedown_vector = y.tolist()

    # create a result dictionary to hold the combination of word and its new vector
    result_vectors = dict()
    for i in range(len(word_list)):
        result_vectors[word_list[i]] = sizedown_vector[i]

    with open(str(output_path), 'w') as fp:
        json.dump(result_vectors, fp, sort_keys=True, indent=4)

    with open(str(output_path_csv), 'w', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result_vectors.items():
            writer.writerow([key.encode('ascii', 'ignore').decode('utf8'), value[0], value[1]])

    x_coords = y[:, 0]
    y_coords = y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_list, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig(str(output_img))


def main(argv):
    try:
        options, args = getopt.getopt(argv, 'f:', ['file_path='])
        if len(options) == 0 or len(options) > 1:
            print('usage : tsne_reduction.py -f <path to word2vec result in json format>')
            sys.exit(2)
        else:
            for flag, arg in options:
                if '-f' in flag or '--file_path' in flag:
                    file_path = Path(arg)
                    if not file_path.exists():
                        assert False, 'file doesn\'t exist'
                else:
                    assert False, 'unexpected argument'
            tsne_reduction(file_path)
    except getopt.GetoptError:
        print('usage : tsne_reduction.py -f <file_path>')
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
