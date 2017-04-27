# coding=utf-8
"""naiveBayes.py

使用朴素贝叶斯进行模型预测
P(C|x1,x2,x3....xn)正比于P(C)*P(x1|C)*P(x2|C)*P(x3|C)*...*P(xn|C)


"""
from toolFunc import segText,rmvStopWord
import cPickle
import math
from numpy import rate
import jieba
def countRate(wl,cls2Text):
    """预测文本对类别的概率
    
    参数:
        wl            文本对应词列表
        cls2Text 类别-Text词典  形式{class1:[Text1,Text2...],class2:[...]}
    返回
        最大概率标签
    """
    maxRate=-float('inf')
    label=None
    for cls in cls2Text.keys():
        rate=0.0
        for word in wl:
            num=2
            for text in cls2Text[cls]:
                if word in text:
                    num+=1
            rate+=math.log(float(num)/len(cls2Text[cls]))
        if rate>maxRate:
            maxRate=rate
            label=cls
    return label
    

def predict(text,cls2Text):
    """预测文本所属标签
    
    参数:
    text            输入文本
    
    返回:
        预测标签
    """

    wordList=rmvStopWord(list(set(segText(text))))
    wl=[]
    for i in wordList:
        if len(i)>3:
            wl.append(i)
    
    return countRate(wl,cls2Text)
    


if __name__=='__main__':
    jieba.load_userdict('dict.txt')
    cls2Text = cPickle.load(open("model/cls2Text.pkl", 'r'))  # 形式{class1:[Text1,Text2...],class2:[...]}
    test = open('dataSet/test.txt', 'r')
    errorNum=0
    allNum=0
    for line in test:
        allNum+=1
        truelabel=line.split(' ')[0]
        text=line.split(' ')[1]
        predictLable=predict(text,cls2Text)
        print truelabel,predictLable
        if (predictLable!=truelabel):
            errorNum+=1

    cPickle.dump(errorNum, open("errornum.pkl",'w'), protocol=0)
    print  "acc=%.2f%%"%(float(allNum-errorNum)*100/allNum)


