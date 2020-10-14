#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2020/6/23 14:00
# @Author : SN
# @Version：V 0.1
# @File : run.py
# @desc :

from PMRank.PMRank import PMRank


def run():
    text = '''There is a park near my home. There are a lot of beautiful trees, flowers and birds in the park. So many people go to the park to enjoy their weekends. They like walking or having a picnic in the park. But I like flying a kite with my sisiter there.'''
    keyphrases = PMRank(text, 10)
    for keyphrase in keyphrases:
        print(keyphrase)


if __name__ == '__main__':
    run()


