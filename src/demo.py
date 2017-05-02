#coding=utf-8
import gensim.models as mod
import cPickle
import gensim.models as vecMod
from pylab import *
from RNNs import RNNs
# 导入模型
# def getclose(model,str):
#     result = model.most_similar(str)
#     for each in result:
#         print each[0] , each[1]
# model=mod.Word2Vec.load('model/wordVec.model')
# model[u'金融']
# getclose(model,u'金融')
figure(figsize=(8,6),dpi=80)
cnn=RNNs()