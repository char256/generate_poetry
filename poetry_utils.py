# -*- coding: utf-8 -*-
import numpy as np
import types
import cn_utils

input_dim = 20
max_word = 100

def _generate_feature(poetry):
    assert type(poetry.title[0]) == types.IntType
    feature = np.zeros(input_dim)
    #feature[0:len(poetry.title)] = poetry.title
    lenth = len(poetry.title)
    for i in xrange(int(input_dim/(lenth+1))):
        feature[i*(lenth+1): i*(lenth+1)+lenth] = poetry.title
    print "poetry_utils._generate_feature:", feature
    return feature


def generate_data(poetries, data, split = 'train'):
    '''
    输出的data里，data[feature]是题目的编码，一共20个整数，不足
    补0。data[captions]是古诗的内容，一共100个字符，不足补0
    '''
    assert type(poetries[0].title[0]) == types.IntType
    assert split == 'train' or split == 'val'
    N = len(poetries)
    D = input_dim
    T = max_word
    features = np.zeros((N,D))
    captions = np.zeros((N,T))
    for i in xrange(N):
        pt = poetries[i]
        feature = _generate_feature(pt)
        features[i,:] = feature
        captions[i,:len(pt.content)] = pt.content
    
    data['%s_features' % split] = features
    data['%s_captions' % split] = captions
    return data

def title_to_feature(title, word_to_idx):
    feature = np.zeros((1,input_dim))
    assert len(title) <= 60, 'the length of title should not be longer than 20 words!'

    caption = np.array(cn_utils.encodeChinese(title, word_to_idx))
    feature[0, :caption.shape[0]] = caption
    return feature