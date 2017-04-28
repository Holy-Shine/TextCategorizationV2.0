#coding=utf-8
import gensim.models as mod
import cPickle
import gensim.models as vecMod
# 导入模型
def getclose(model,str):
    result = model.most_similar(str)
    for each in result:
        print each[0] , each[1]
model=mod.Word2Vec.load('model/wordVec.model')

class T(object):
    def get(self):
        while 1:
            w2vmod = vecMod.Word2Vec.load('model/wordVec.model')
            with open('dataSet/rnn_train.txt', 'r') as f:
                for line in f:
                    label = [0.0] * 6
                    sample = [[0.0] * 2] * 3
                    arr = line.split('::')
                    if len(arr) > 1:
                        label[int(arr[0]) - 2] = 1.0
                        wordsL = arr[1].split(' ')
                        i = 0
                        for index in xrange(3):
                            if wordsL[index].decode('utf-8') in w2vmod:
                                sample[index] = w2vmod[wordsL[index].decode('utf-8')].tolist()

                    yield  (sample,label)



h=T()
i=h.get()
print next(i)
print i