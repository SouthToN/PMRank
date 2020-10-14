#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/6/23 14:08
# @Author : SN
# @Version：V 0.1
# @File : embedding.py
# @desc : Get vector representation

from sentence_transformers import SentenceTransformer, LoggingHandler
from PMRank.calculate_distance import calculateDistance
import numpy as np
import logging


'''Processing log'''

np.set_printoptions(threshold=100)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()])

'''Load model'''
model = SentenceTransformer('D:/Desktop/myModel/auxiliary_data/bert-base-nli-mean-tokens/')

def get_keyphrase_candidate_enbeddings(keyphrase_candidate):
    """Obtain candidate word vectors according to the pre-trained model"""
    if len(keyphrase_candidate) > 0:
        keyphrase_candidate_enbedding = model.encode(keyphrase_candidate)
    return keyphrase_candidate_enbedding


def get_sentence_enbeddings(sents_sectioned):
    """Get the vector of each sentence in the document according to the pre-trained model"""
    if len(sents_sectioned) > 0:
        sentence_embedding = model.encode(sents_sectioned)
    return sentence_embedding


'''********************examples***********************'''


def sent_embedding_demo():
    # Embed a list of sentences
    # sentences = ['This framework generates embeddings for each input sentence',
    #              'Sentences are passed as a list of string.',
    #              'The quick brown fox jumps over the lazy dog.']
    sentences1 = ['There is a park near my home. There are a lot of beautiful trees, flowers and birds in the park. ']
    sentences2 = ['banana']
    sentences3 = ['apple']
    sentences4 = ['park']
    sentences5 = ['tree']
    sentence_embeddings1 = model.encode(sentences1)
    sentence_embeddings2 = model.encode(sentences2)
    sentence_embeddings3 = model.encode(sentences3)
    sentence_embeddings4 = model.encode(sentences4)
    sentence_embeddings5 = model.encode(sentences5)
    embeddings1=0
    embeddings2=0
    embeddings3=0
    embeddings4=0
    embeddings5=0


    # The result is a list of sentence embeddings as numpy arrays
    for sentence, embedding in zip(sentences1, sentence_embeddings1):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape)
        embeddings1 = embedding

    for sentence, embedding in zip(sentences2, sentence_embeddings2):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape)
        embeddings2 = embedding

    for sentence, embedding in zip(sentences3, sentence_embeddings3):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape)
        embeddings3 = np.array(embedding)
        print("")
    for sentence, embedding in zip(sentences4, sentence_embeddings4):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape)
        embeddings4 = np.array(embedding)
        print("")
    for sentence, embedding in zip(sentences5, sentence_embeddings5):
        print("Sentence:", sentence)
        print("Embedding:", embedding.shape)
        embeddings5 = np.array(embedding)
        print("")

    distance = calculateDistance
    sentences1_sentences2 = distance.get_cos_sim(embeddings1, embeddings2)
    sentences1_sentences3 = distance.get_cos_sim(embeddings1, embeddings3)
    sentences1_sentences4 = distance.get_cos_sim(embeddings1, embeddings4)
    sentences1_sentences5 = distance.get_cos_sim(embeddings1, embeddings5)
    # sentences2_sentences4 = distance.get_cos_sim(embeddings2, embeddings4)
    # sentences3_sentences4 = distance.get_cos_sim(embeddings3, embeddings4)

    print("sentences1-sentences2:", sentences1_sentences2)
    print("sentences1-sentences3:", sentences1_sentences3)
    print("sentences1-sentences4:", sentences1_sentences4)
    print("sentences1-sentences5:", sentences1_sentences5)
    # print("sentences2-sentences4:", sentences2_sentences4)
    # print("sentences3-sentences4:", sentences3_sentences4)


def example():
    print()


if __name__ == '__main__':
    sent_embedding_demo()

'''************************************************'''

