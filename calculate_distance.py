#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/6/21 19:13
# @Author : SN
# @Version：V 0.1
# @File : calculate_distance.py
# @desc :Calculate distance

import nltk
from nltk.corpus import stopwords
import numpy as np

wnl = nltk.WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


class calculateDistance():

    def get_cos_sim(vector_a, vector_b):
        """
        Calculate the cosine similarity between two vectors
        :param vector_a: vector a
        :param vector_b: vector b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denom == 0.0:
            return 0.0
        else:
            cos = num / denom
            sim = 0.5 + 0.5 * cos
            return sim



