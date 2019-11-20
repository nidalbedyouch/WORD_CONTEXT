# coding: utf-8

# A script to do work2vec, taking the raw text data
# Using Gensim
# initialize model with gensim object
# https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/
# ONCE we have vectors
# step 3 - build model
# 3 main tasks that vectors help with
# DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
# more dimensions, more computationally expensive to train
# but also more accurate
# more dimensions = more generalized
import getopt
import multiprocessing
import json
import os
import sys
import time
import pathlib
from WORD_CONTEXT.pre_process import pre_process as prp
import gensim.models.word2vec as w2v


def build(file_path, size, min, window, stop_list):
    sentences = []
    with open(str(file_path), encoding="utf-8") as f:
        for line in f:
            sentence = prp.sentence_to_word_list(line, stop_list=stop_list)
            if len(sentence) > 0:
                sentences.append(sentence)
    output_path = file_path.parent / file_path.stem
    if not output_path.exists():
        output_path.mkdir(parents=True)
    build_model_word2vec(sentences, size, min, window, stop_list, output_path)


def build_model_word2vec(sentences, size, min, window, stop_list, output_path):
    num_features = size
    # Minimum word count threshold.
    min_word_count = min

    # Number of threads to run in parallel.
    # more workers, faster we train
    num_workers = multiprocessing.cpu_count()

    # Context window length.
    context_size = window

    # Downsample setting for frequent words.
    # 0 - 1e-5 is good for this
    downsampling = 1e-3

    # Seed for the RNG, to make the results reproducible.
    # random number generator
    # deterministic, good for debugging
    seed = 1
    # sg – This defines the algorithm. If equal to 1, the skip-gram technique is used. Else, the CBoW method is employed
    sg = 1
    start = time.time()

    model = w2v.Word2Vec(
        sg=sg,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print('Model built ' + str((time.time() - start)) + ' secs')

    word_vectors = dict()

    for sentence in sentences:
        for word in sentence:
            try:
                # The word vectors are stored in a KeyedVectors instance in model.wv.
                # .tolist() -- Converting NumPy array into Python List structure
                word_vectors[word] = list(model.wv[word].tolist())
            except KeyError:
                break

    if not output_path.exists():
        output_path.mkdir(parents=True)
    with open(str(output_path / "gensim_result.json"), 'w') as fp:
        json.dump(word_vectors, fp, sort_keys=True)

    print('End')

    # save model in binary format
    model.wv.save_word2vec_format(str(output_path / 'model.bin'), binary=True)
    prp.word_cloud(sentences, stop_list, output_path)


def main(argv):
    try:
        options, args = getopt.getopt(argv, 'f:s:m:w:',
                                      ['file_path=', 'size', 'min_word_count', 'window'])
        if len(options) == 0 or len(options) > 4:
            print('usage: w2v_model.py -f <file_path> -s <size> -m <min word count> -w <window>')
            sys.exit(2)
        else:
            size = 300
            min = 3
            window = 7
            for flag, arg in options:
                if '-f' in flag or '--file_path' in flag:
                    file_path = pathlib.Path(arg)
                    print(file_path)
                    if not file_path.exists():
                        assert False, 'File doesn\'t exist'
                elif '-s' in flag or '--size' in flag:
                    size = int(arg)
                elif '-m' in flag or '--min_word_count' in flag:
                    min = int(arg)
                elif 'w' in flag or '--window' in flag:
                    window = int(arg)
                else:
                    assert False, 'unexpected argument'
        stopwords_path = pathlib.Path(__file__).absolute().parent.parent.parent / pathlib.Path('resources/stopwords.json')
        stop_list = json.loads(open(str(stopwords_path)).read())
        build(file_path=file_path, size=size, min=min, window=window, stop_list=stop_list)
    except getopt.GetoptError:
        print('usage: w2v_model.py -f <file_path> -s <size> -m <min word count> -w <window>')
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])

#############################
#       Usage Examples      #
#############################


# build_model_word2vec(file_name,False,[],lemmatiser=False)
# build_model_word2vec(file_name,True,['NOM','ADJ','NAM','VER'],lemmatiser=False)
# build_model_word2vec(file_name,annoter=True,keep=['NOM','ADJ','NAM','VER'],lemmatiser=True)
# build_model_word2vec(file_name,True,['NOM','NAM','ADJ'],lemmatiser=True)
# build_model_word2vec(file_name,True,['NOM','ADJ','NAM'],lemmatiser=False)

# print (len(model.wv.vocab))
# toprint =model.wv.similarity('algorithme', u'méthode')
# print 'similitude entre algorithme et méthode : '+ str(toprint)
# toprint =model.wv.similarity('cluster', 'classifier')
# print 'similitude entre cluster et classifier : '+ str(toprint)
# ms = model.wv.most_similar(positive = ['apprentissage','cluster'],negative=['public'])
# print 'mot plus proche de apprentissage et cluster et loin de public'
# for k,v in ms:
#     print '     '+k
# ms = model.wv.most_similar(positive = ['twitter',u'développement'],negative=[])
# print(ms)
# dm = model.wv.doesnt_match(['twitter' ,'tweets' ,'hashtags', 'sociaux'])
# print(dm)
