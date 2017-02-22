# -*- coding: utf-8 -*-
import os
from Poetry import Poetry
import types

start_caption = 'stt'
end_caption = 'end'
blank_caption = 'blk'

def readFile(filepath):
    """
    中文单词的编码是三个一组，每次读取后有一个换行，为了以后处理方便，
    在读取文件的时候直接将换行删除
    """
    filename= os.listdir(filepath)
    pt = []
    print 'all files:'
    print filename
    for f in filename:
        fullname = os.path.join(filepath,f)
        print 'reading file: ', fullname
        fopen = open(fullname, 'r')
        line = 0
        title = ''
        content = ''
        for eachLine in fopen:
            if line == 0:
                title = eachLine[:-1]
            else:
                content = content + eachLine
                if content[-1] == '\n':
                    content = content[:-1]
            line += 1

        content = start_caption + content + end_caption
        pt.append(Poetry(title = title,content = content))
        print title
        print content
        fopen.close()
        print 5*'-', 'reading finished', 5*'-'
        print '\n'
    return pt


def get_encoding(word_to_idx, idx_to_word, letters):
    '''
    每3个一组进行汉字的编码
    '''
    tot = len(word_to_idx)
    assert len(letters) % 3 == 0, 'Chinese chracters are coded 3 in each group!'
    for i in xrange(0,len(letters)-1,3):
        ch = letters[i] + letters[i+1] + letters[i+2]
        if word_to_idx.has_key(ch): pass
        else :
            idx_to_word.append(ch)
            word_to_idx[ch] = int(tot)
            tot += 1
    return word_to_idx, idx_to_word


def get_idx_and_words(poetries):
    '''
    从训练数据中得到编码器word_to_idx和解码器idx_to_word
    '''
    word_to_idx = {}
    idx_to_word = []
    word_to_idx[blank_caption] = int(0)
    idx_to_word.append(blank_caption)
    word_to_idx[start_caption] = int(1)
    idx_to_word.append(start_caption)
    word_to_idx[end_caption] = int(2)
    idx_to_word.append(end_caption)
    for pt in poetries:
        word_to_idx, idx_to_word = get_encoding(word_to_idx, idx_to_word, pt.title)
        word_to_idx, idx_to_word = get_encoding(word_to_idx, idx_to_word,pt.content)
    return word_to_idx, idx_to_word


def encodeChinese(letters, word_to_idx):
    '''
    将一句话进行编码
    '''
    assert len(letters) % 3 == 0
    captions = []
    for i in xrange(0,len(letters), 3):
        ch = letters[i] + letters[i+1] + letters[i+2]
        captions.append(word_to_idx[ch])
    return captions

def encodePoetry(poetry, word_to_idx):
    '''
    将一首诗进行编码
    '''
    encode_poetry = Poetry()
    encode_poetry.title = encodeChinese(poetry.title, word_to_idx)
    encode_poetry.content = encodeChinese(poetry.content, word_to_idx)
    return encode_poetry

def decodeChinese(captions, idx_to_word):
    '''
    对一句话进行解码
    '''
    decoded = ''
    for c in captions:
        decoded += idx_to_word[c]
    return decoded

def decodePoetry(encoded_poetry, idx_to_word):
    '''
    对一首诗进行解码
    '''
    title = encoded_poetry.title
    poetry_captions = encoded_poetry.content
    if type(title[0]) == types.IntType:
        title = decodeChinese(title, idx_to_word)
    content = decodeChinese(poetry_captions, idx_to_word)
    return Poetry(title = title, content = content)

