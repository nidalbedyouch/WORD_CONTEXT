import matplotlib.pyplot as plt;
import networkx as nx;
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from WORD_CONTEXT.pre_process.pre_process import tag_and_remove_as_text


def tf_idf(filepath, stop_list):
    docs = pd.read_json(filepath)
    docs = docs[:50]
    print(docs)
    # clean text documents
    for x in docs.index:
        text = tag_and_remove_as_text(docs.iloc[x]['text'], stoplist=stop_list)
        docs.loc[x, 'text'] = text
    # print docs.head()
    # comput tfidf of words
    tfidf = TfidfVectorizer(stop_words=stop_list)
    tfs = tfidf.fit_transform(docs['text'])
    print(list(tfidf.vocabulary_.keys())[:10] + ['...'])

    feature_names = tfidf.get_feature_names()
    corpus_index = [n for n in list(tfidf.vocabulary_.keys())]
    df = pd.DataFrame(tfs.todense(), columns=feature_names)
    print(df)
    # get top 30 words based on tfidf
    top_words = []
    for x in range(0, len(df)):
        row = df.iloc[x]
        row = row.sort_values(ascending=False)[:100]
        top_word = []
        for i, v in row.items():
            top_word.append(i)
        top_words.append(top_word)
    # linking top_words to document id
    top_word_frame = pd.DataFrame(list(zip(docs['id'], top_words)),
                                  columns=['id', 'top_words'])

    print(top_word_frame)
    # for x in df["apprentissage"].sort_values(ascending=False).head(n=10).index:
    #     print(docs.iloc[x]["id"])

    return top_word_frame


def compute_classes(vectors):
    dict_classe = dict()
    for x in vectors:
        if dict_classe.has_key(x):
            dict_classe[x] += 1
        else:
            dict_classe[x] = 1
    return sorted(dict_classe.items(), key=lambda x: x[1], reverse=True)[:5]

def doc_classification(top_words_frame, kmeans_clustering, model):
    clusters = []
    for x in top_words_frame.index:
        vectors = []
        for word in top_words_frame.loc[x, 'top_words']:
            try:
                vectors.append(model[word])
            except KeyError:
                print('Word not in model')
                pass
        clusters.append(compute_classes(kmeans_clustering.predict(vectors)))

    dict_words = {'id': top_words_frame['id'], 'classe': clusters}
    classes = pd.DataFrame(dict_words)
    print(classes)
    return classes


def network(classes, centers):
    G = nx.DiGraph()
    for row in classes.itertuples():
        print(row)
        # G.add_node(row.id,nodesize = len(str(row.id)))
        for k, v in row.classe:
            print(centers[k])
            to = " ".join(centers[k])
            # G.add_node(to,nodesize = len(to))
            G.add_edge(row.id,to,title=str(v))
    labels = nx.get_edge_attributes(G, 'title')
    plt.figure()
    # sizes = [G.node[node]['nodesize'] * 200 for node in G]
    """
    Using the spring layout : 
    - k controls the distance between the nodes and varies between 0 and 1
    - iterations is the number of times simulated annealing is run
    default k=0.1 and iterations=50
    """
    pos = nx.spring_layout(G)
    nx.draw(G, node_size=800,pos=pos,node_shape="s",with_labels=True)
    nx.draw_networkx_edge_labels(G,pos, edge_labels=labels)
    plt.show()
