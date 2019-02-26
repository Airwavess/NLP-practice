#!/usr/bin/env python
# coding: utf-8

# In[142]:


import json
import re


# In[143]:


DATASET_DIR = './WebNews.json'
with open(DATASET_DIR) as f:
    dataset = json.load(f)


# In[144]:


seg_list = list(map(lambda d: d['detailcontent'], dataset))


# In[145]:


rule = re.compile(r"[^\u4e00-\u9fa5]")
line = rule.sub('', ''.join(seg_list))


# In[147]:


# coding: utf-8
from collections import Counter, namedtuple
import re
import pandas as pd
import numpy as np

def ngram(data, N=2):
    
    split_words = list(data)
    
    # 計算分子
    total_grams = list()
    [total_grams.append(tuple(split_words[i:i+N])) for i in range(len(split_words)-N+1)]
    # 計算分母
    words = list()
    [words.append(tuple(split_words[i:i+N-1])) for i in range(len(split_words)-N+2)]

        
    ngram_prediction = dict()
    total_word_counter = Counter(total_grams)
    word_counter = Counter(words)
    
    Word = namedtuple('Word', ['word', 'prob'])
    for key in total_word_counter:
        word = ''.join(key[:N-1])
        if word not in ngram_prediction:
            ngram_prediction.update({word: []})
        
        w = Word(key[-1], '{:.3g}'.format(total_word_counter[key]/word_counter[key[:N-1]]))
        ngram_prediction[word].append(w)
        
    return ngram_prediction


# In[148]:


# unigram = ngram(line, N=1)

# bigram_prediction = ngram(line, N=2)

# for ng in bigram_prediction.values():
#     ng.sort(key=lambda x: x.prob, reverse=True)

tri_prediction = ngram(line, N=3)

for ng in tri_prediction.values():
    ng.sort(key=lambda x: x.prob, reverse=True)


# In[149]:


corpus = '韓國'
corpus_len = 25
for i in range(corpus_len):
    last_word = corpus[-2:]
    next_word = tri_prediction[last_word][0].word
    corpus += next_word


# In[150]:


corpus

