#coding=utf-8
import gensim.models as mod
# 导入模型
model=mod.KeyedVectors.load_word2vec_format('model/wordVec.vector',binary=False)
print model[u'金融学'.encode('utf-8')]