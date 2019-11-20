# coding : utf-8
import os
import treetaggerwrapper
import unicodedata
import numpy as np
import matplotlib.pyplot as plt

# put directory to tree tagger
dir_tree_tagger = '/home/etudiant/TreeTagger'


def build_tree_tagger(text, source_file, output_path):
    global dir_tree_tagger
    # build a TreeTagger wrapper
    tagger = treetaggerwrapper.TreeTagger(TAGDIR=dir_tree_tagger, TAGLANG="fr")
    # tag text
    tags = tagger.tag_text(text)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    treetaggerwrapper.TreeTagger.tag_file_to(tagger, str(source_file), str(output_path / 'tagger_result.txt'))
    # pprint.pprint(tags)
    tags2 = treetaggerwrapper.make_tags(tags)
    # pprint.pprint(tags2)
    tag_dict = dict()
    for tag in tags2:
        if hasattr(tag, 'pos'):
            tag_dict[unicodedata.normalize('NFD', tag.word).encode('ascii', 'ignore')] = {"pos": tag.pos,
                                                                                              "lemma": tag.lemma}
    # pprint.pprint(tags2)
    return tag_dict, tags2


def take(n, donnee):
    "Return first n items of the iterable as a list"
    r = []
    i = 0
    for key, value in sorted(donnee.items(), key=lambda x: x[1], reverse=True):
        r.append((key, value))
        # i += 1
        # if i >= n:
        #     break
    return r


def build_dict(tags):
    donnee = dict()
    # count words in each tag existing
    for tag in tags:
        if hasattr(tag, 'pos'):
            pos = tag.pos
            if pos.startswith('VER'):
                # print "verb : " + tag.word + " " + pos
                pos = 'VER'
            elif pos.startswith('DET'):
                pos = 'DET'
            if pos in donnee:
                donnee[pos] += 1
            else:
                donnee[pos] = 1
    return donnee


def visualise(donnee, img, output_path):
    # sort and pick first 5 tags
    n_items = take(5, donnee)

    # plot
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

    data = [x[1] for x in n_items]
    tag = [x[0] for x in n_items]

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}%\n({:d} )".format(pct, absolute)

    wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"))

    ax.legend(wedges, tag,
              title="Tag",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")

    ax.set_title("Tags Pie " + str(np.sum(data)))
    plt.savefig(str(output_path / img))
    #plt.show()

# dta, tagger = build_tree_tagger(source_text)
