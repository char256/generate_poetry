import numpy as np
import types
import cn_utils

input_dim = 20
max_word = 100

def generate_feature(poetry):
    assert type(poetry.title[0]) == types.IntType
    feature = np.zeros(input_dim)
    feature[0:len(poetry.title)] = poetry.title
    return feature


def generate_data(poetries, data, split = 'train'):
    assert type(poetries[0].title[0]) == types.IntType
    assert split == 'train' or split == 'val'
    N = len(poetries)
    D = input_dim
    T = max_word
    features = np.zeros((N,D))
    captions = np.zeros((N,T))
    for i in xrange(N):
        pt = poetries[i]
        feature = generate_feature(pt)
        features[i,:] = feature
        captions[i,:len(pt.content)] = pt.content
    
    data['%s_features' % split] = features
    data['%s_captions' % split] = captions
    return data
