#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import logging

output = open('/Users/Vicky/Desktop/dist/vec.txt','a+')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        
        flipped = {}
        
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


sources = { '/Users/Vicky/Desktop/dist/D.txt':'TRAIN_NEG', '/Users/Vicky/Desktop/dist/ND.txt':'TRAIN_POS'}

sentences = LabeledLineSentence(sources)

model = Doc2Vec(min_count=1, window=15, size=130, sample=1e-4, negative=5, workers=8)
model.build_vocab(sentences.to_array())

for epoch in range(10):
    model.train(sentences.sentences_perm(),total_examples=model.corpus_count,epochs=model.iter)

train_arrays = numpy.zeros((1058, 130))
train_labels =[]

y=[]

with open('/Users/Vicky/Desktop/ComParE2016_Deception/dist/ComParE2016_Deception.tsv') as f:
    for line in f.readlines():
        lists = line.split(',')
        if lists[1]=='D':
            y.append(lists[0])
f.close()

with open('/Users/Vicky/Desktop/ComParE2016_Deception/dist/ComParE2016_Deception.tsv') as f:
    for line in f.readlines():
        lists = line.split(',')
        if lists[1]=='ND':
            y.append(lists[0])
f.close()

for i in range(1058):
    if(i<311):
        prefix_train_neg = 'TRAIN_NEG_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_neg]
        train_labels.append('D')
    # print len(train_arrays[i])
        print>>output, y[i], train_labels[i], train_arrays[i]
    if(i>=311 and i<1058):
        j=i-311
        prefix_train_pos = 'TRAIN_POS_' + str(j)
        train_arrays[i]=model.docvecs[prefix_train_pos]
        train_labels.append('ND')
#   print len(train_arrays[i])
        print>>output, y[i], train_labels[i], train_arrays[i]

output.close()
