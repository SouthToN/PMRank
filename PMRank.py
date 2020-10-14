#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/6/23 13:58
# @Author : SN
# @Version：V 0.1
# @File : PMRank.py
# @desc :

from stanfordcorenlp import StanfordCoreNLP
from PMRank.calculate_distance import calculateDistance
from PMRank.embedding import get_keyphrase_candidate_enbeddings, get_sentence_enbeddings
import re
import nltk
from nltk.corpus import stopwords
stopword_dict = set(stopwords.words('english'))

GRAMMAR = """  NP: {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

def PMRank(text, N=15):
    """
        :param text:  documnet
        :param N:  Number of keyphrases to be extracted
    """
    stanfordCoreNLP = StanfordCoreNLP(r'D:\Desktop\stanford-corenlp-latest\stanford-corenlp-4.0.0', quiet=True)
    distance = calculateDistance
    sents_sectioned = get_sent_segmented(text)
    tokens = stanfordCoreNLP.word_tokenize(text)
    tokens_tagged = stanfordCoreNLP.pos_tag(text)
    assert len(tokens) == len(tokens_tagged)
    for i, token in enumerate(tokens):
        if token.lower() in stopword_dict:
            tokens_tagged[i] = (token, "IN")
    keyphrase_candidate = extract_candidates(tokens_tagged)
    candidate_embeddings_list = get_keyphrase_candidate_enbeddings(keyphrase_candidate)
    sentence_enbeddings_list = get_sentence_enbeddings(sents_sectioned)

    dist_list = {}  # {'keyphrase', score}
    for i, wemb in enumerate(candidate_embeddings_list):
        sum = 0
        for j, semb in enumerate(sentence_enbeddings_list):
            sum += distance.get_cos_sim(wemb, semb)
        dist = sum/len(sentence_enbeddings_list)
        if keyphrase_candidate[i] in dist_list:
            dist_list[keyphrase_candidate[i]] = dist
        else:
            dist_list[keyphrase_candidate[i]] = []
            dist_list[keyphrase_candidate[i]] = dist

    dist_sorted = sorted(dist_list.items(), key=lambda x: x[1], reverse=True)
    return dist_sorted[0:N]


def get_sent_segmented(text):
    """token"""
    sents_sectioned = []
    tokens = nltk.tokenize.sent_tokenize(text)
    for i, token in enumerate(tokens):
        sent = str(token)
        sent = re.sub(u'\u2002', " ", sent)
        sents_sectioned.append(sent)
    return sents_sectioned


def extract_candidates(tokens_tagged):
    """
    desc:extract candidates
    :param tokens_tagged:Token after part of speech tagging
    :return a list of candidates: ['candidates_word1','candidates_word2','candidates_word3',...]
    """
    np_parser = nltk.RegexpParser(GRAMMAR)
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if isinstance(token, nltk.tree.Tree) and token._label == "NP":
            np = ' '.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            count += length
            keyphrase_candidate.append(np)
        else:
            count += 1
    return keyphrase_candidate
