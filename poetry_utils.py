import numpy as np
import types
import cn_utils

input_dim = 20
max_word = 100

def _generate_feature(poetry):
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