# coding: utf-8

import codecs
import collections
# A script to clean data
import getopt
import json
import os
import re
import sys
import time

import langid
import matplotlib.pyplot as plt
import nltk
import unicodedata
from pathlib import Path
from wordcloud import WordCloud
from str2bool import str2bool
from WORD_CONTEXT.pre_process import tag_util as tg

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')


# function to compute words
def count_words(text):
    n = 0
    for line in text:
        n += len(line)
    return n


def word_list_to_text(sentences):
    text = u""
    for sentence in sentences:
        text += ' '.join(sentence) + "\n"
    return text


# cleaning data
def sentence_to_word_list(raw, stop_list):
    # remove non letters and split into words
    # ^[a-zA-Z] means any a-z or A-Z at the start of a line
    # [^a-zA-Z] means any character that IS NOT a-z OR A-Z, it will be replaced by " " in this case
    # re: regular expression library

    # special escaping character '...'
    raw = raw.replace(u'\u2026', '.')
    raw = raw.replace(u'\u00a0', ' ')
    raw = raw.replace(u'\n', ' ')
    # remove accent btw
    raw = unicodedata.normalize('NFD', raw).encode('ascii', 'ignore')
    make_sense = re.sub("[^a-zA-Z]", " ", raw.decode("utf-8"))
    real_words = make_sense.lower().split()

    # Create a list of words
    texts = [word for word in real_words if word not in stop_list and len(word) > 2]
    return texts


def remove_stopwords(raw_sentences, stop_list):
    sentences = []
    print('Removing stop words ...')
    start_remove = time.time()
    for raw in raw_sentences:
        if len(raw) > 0:
            # A list within list
            sentences.append(sentence_to_word_list(raw, stop_list))
    print('Stop words removed spent ' + str((time.time() - start_remove)) + ' secs')
    return sentences


def clean_corpus_txt(file_path, output_path, stop_list, annotate, keep, lem):
    source_text = u""
    with codecs.open(str(file_path), "r", "utf-8") as raw:
        text = raw.read()
        classify = langid.classify(text)
        if classify[0] == 'fr':
            source_text += text
    return clean(source_text=source_text, file_path=file_path, output_path=output_path, stop_list=stop_list,
                 annotate=annotate, keep=keep, lem=lem)


def clean_corpus_json(source_path, output_path, stop_list, annotate, keep, lem):
    content = json.loads(open(str(source_path),encoding="utf-8").read())
    source_text = u""
    for doc in content:
        # print doc['id']
        classify = langid.classify(doc['text'])
        if classify[0] == 'fr':
            source_text += doc['text']
    return clean(source_text=source_text, file_path=source_path, output_path=output_path, stop_list=stop_list,
                 annotate=annotate, keep=keep, lem=lem)


def clean_word(word):
    special = ["&", "#", "~", "{", "[", "(", "|", "|", "]", ")", "}", "=", "+", "*"]
    table = {ord(char): None for char in special}
    pos = word.find("|")
    if pos != 0 and pos != len(word) - 1:
        word = word.split('|')[0]
    word = word.translate(table)
    return word.lower()


def tag_and_remove_as_text(source_text, stop_word_list, keep, lem):
    print('Tagging  ...')
    start_remove = time.time()
    text = u""
    tag_dict, tags2 = tg.build_tree_tagger(source_text)
    i = 0
    while i < len(tags2):
        tag = tags2[i]
        if hasattr(tag, 'pos'):
            if lem:
                word = tag.lemma
            else:
                word = tag.word
            word = clean_word(word)
            if any(tag.pos.startswith(w) for w in keep) and len(word) > 2 and word.isalpha() \
                    and word not in stop_word_list:
                text += " " + word
        i += 1
    print('tagger built :  ' + str((time.time() - start_remove)) + ' secs')
    return text


def tag_and_remove(source_text, file_path, output_path, stop_word_list, keep):
    print('Tagging  ...')
    start_remove = time.time()
    text = []
    tag_dict, tags2 = tg.build_tree_tagger(source_text, file_path, output_path)
    tg.visualise(tg.build_dict(tags2), "tag_before", output_path)
    # to_remove = ['ADV', 'VER', 'DET', 'ABR', 'NUM', 'INT', 'KON', 'PRO', 'PRP', 'SYM', 'SENT', 'PUN']

    i = 0
    data = dict()
    while i < len(tags2):
        tag = tags2[i]
        sentence = []
        while hasattr(tag, 'pos') and tag.pos != 'SENT' and i < len(tags2):
            word = tag.word
            word = clean_word(word)
            if any(tag.pos.startswith(w) for w in keep) and len(
                    word) > 2 and word.isalpha() and word not in stop_word_list:
                sentence.append(word)
                # add to data to display
                if tag.pos.startswith('VER'):
                    # print "verb : " + tag.word + " " + pos
                    pos = 'VER'
                elif tag.pos.startswith('DET'):
                    pos = 'DET'
                else:
                    pos = tag.pos
                if pos in data:
                    data[pos] += 1
                else:
                    data[pos] = 1
            i += 1
            if i < len(tags2):
                tag = tags2[i]
        tags2.remove(tag)
        text.append(sentence)
        i += 1

    # view left tags and their frequency
    tg.visualise(data, "tag_after", output_path)
    print('tagger built :  ' + str((time.time() - start_remove)) + ' secs')
    return text


def tag_remove_lem(source_text, file_path, output_path, stop_word_list, keep):
    print('Tagging  ...')
    start_remove = time.time()
    text = []
    tag_dict, tags2 = tg.build_tree_tagger(source_text, file_path, output_path)
    tg.visualise(tg.build_dict(tags2), "tag_before", output_path)
    # to_remove = ['ADV', 'VER', 'DET', 'ABR', 'NUM', 'INT', 'KON', 'PRO', 'PRP', 'SYM', 'SENT', 'PUN']
    i = 0
    print(len(tags2))
    data = dict()
    while i < len(tags2):
        tag = tags2[i]
        sentence = []
        while hasattr(tag, 'pos') and tag.word != '.' and i < len(tags2):
            word = tag.lemma
            word = clean_word(word)
            if any(tag.pos.startswith(w) for w in keep) and len(word) > 2 and word.isalpha() \
                    and word not in stop_word_list:
                sentence.append(word)
                # add to data to display
                if tag.pos.startswith('VER'):
                    # print "verb : " + tag.word + " " + pos
                    pos = 'VER'
                elif tag.pos.startswith('DET'):
                    pos = 'DET'
                else:
                    pos = tag.pos
                if pos in data:
                    data[pos] += 1
                else:
                    data[pos] = 1
            i += 1
            if i < len(tags2):
                tag = tags2[i]
        tags2.remove(tag)
        text.append(sentence)
        i += 1
    print(len(tags2))
    # view left tags and their frequency
    tg.visualise(data, "tag_after", output_path)
    print('tagger built :  ' + str((time.time() - start_remove)) + ' secs')
    return text


def clean(source_text, file_path, output_path, stop_list, annotate, keep, lem=True):
    print('Start cleaning text...')
    start_cleaning = time.time()
    # using nltk(natural language toolkit)
    tokenizer = nltk.data.load("tokenizers/punkt/french.pickle")
    sentences = []
    if annotate:
        print('texte annoté')
        # build tagger and only keep the one in the list keep
        if lem:
            print("utilsation des lemmatisation")
            sentences = tag_remove_lem(source_text=source_text, file_path=file_path, output_path=output_path,
                                       stop_word_list=stop_list, keep=keep)
        else:
            print('non utilsation des lemmatisation')
            sentences = tag_and_remove(source_text=source_text, file_path=file_path,
                                       output_path=output_path, stop_word_list=stop_list, keep=keep)
    else:
        print('texte non annoté')
        raw_sentences = tokenizer.tokenize(source_text)
        # remove stop words
        sentences = remove_stopwords(raw_sentences, stop_list)
    print('Data cleaned : ' + str((time.time() - start_cleaning)) + ' secs')
    return sentences


def word_cloud(sentences, stop_list, output_path):
    # build wordCloud
    wordcount = collections.defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            wordcount[word] += 1

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          max_words=2000,
                          stopwords=stop_list,
                          min_font_size=10).generate_from_frequencies(wordcount)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    fn = "wordCloud_" + str(time.time()) + '.jpg'
    plt.savefig(str(output_path / fn))


def process(file_path, annotate=True, keep=["NOM", "VER", "ADJ", "NAM"], lem=True):
    file_name = file_path.stem
    output_path = file_path.parent / file_name
    if not output_path.exists():
        output_path.mkdir(parents=True)
    print(Path(__file__).absolute().parent.parent.parent)
    stopwords_path = Path(__file__).absolute().parent.parent.parent / Path('resources/stopwords.json')
    stop_list = json.loads(open(str(stopwords_path)).read())
    if file_path.suffix in '.txt':
        sentences = clean_corpus_txt(file_path, output_path, stop_list=stop_list, annotate=annotate, keep=keep, lem=lem)
    elif file_path.suffix in '.json':
        sentences = clean_corpus_json(file_path, output_path, stop_list=stop_list,
                                      annotate=annotate, keep=keep, lem=lem)
    else:
        assert False, 'Failed file suffix unknown'
    fn = file_name + "_pre_processed.txt"
    file = open(str(output_path / fn), "w+", encoding='utf-8')
    for s in sentences:
        file.write(' '.join(s) + "\n")
    file.close()
    word_cloud(sentences, stop_list, output_path)


def main(argv):
    use_tree_tagger = True
    lem = True
    tags_to_keep = ["NOM", "VER", "ADJ", "NAM"]
    try:
        options, args = getopt.getopt(argv, 'f:u:k:l',
                                      ['file_path=', 'use_tree_tagger', 'tags_to_keep', 'lem'])
        if len(options) == 0 or len(options) > 5:
            print('usage: pre_process.py -f <file_path> [-u <use_tree_tagger (True or False)>] '
                  '[-k <tags_to_keep example "-k "NOM,VER,ADJ,NAM"> ] [-l <lem (True or False)>]')
            sys.exit(2)
        else:
            for flag, arg in options:
                if '-f' in flag or '--file_path' in flag:
                    file_path = Path(arg)
                    if not file_path.exists():
                        assert False, 'File doesn\'t exist'
                elif '-u' in flag or '--use_tree_tagger' in flag:
                    use_tree_tagger = str2bool(arg)
                elif '-k' in flag or '--tags_to_keep' in flag:
                    keep_list = arg.split(',')
                    for i in keep_list:
                        i.replace(" ", "")
                    allowed = ["NOM", "VER", "ADJ", "NAM", 'ADV', 'VER', 'DET', 'ABR', 'NUM', 'INT', 'KON', 'PRO',
                               'PRP', 'SYM', 'SENT', 'PUN']
                    print(keep_list)
                    inter = set(allowed).intersection(keep_list)
                    if len(inter) > 0:
                        tags_to_keep = inter
                    else:
                        assert False, 'tags choosen aren\'t allowed, list of allowed tags : NOM,VER,ADJ,NAM,ADV,VER,' \
                                      'DET,ABR, NUM, INT, KON, PRO, PRP, SYM, SENT, PUN '
                elif '-l' in flag or '--lem' in flag:
                    lem = str2bool(arg)
                else:
                    assert False, 'unexpected argument'
        process(file_path=file_path, annotate=use_tree_tagger, lem=lem, keep=tags_to_keep)
    except getopt.GetoptError:
        print('usage: pre_process.py -f <file_path> [-u <use_tree_tagger (True or False)>] '
                  '[-k <tags_to_keep example "-k "NOM,VER,ADJ,NAM"> ] [-l <lem (True or False)>]')
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
