# coding=utf-8
"""toolFunc.py

工具函数库

包含模块：
    segText                 中文分词，返回分好词的list
    rmvStopWord    去停用词
"""
import jieba
import re,cPickle
def segText(text):
    """中文分词.
    
    返回一个分好词的中文list
    
    参数：
        text-输入文本
    返回：
        切好词的list
    """
    retList=[]
    seg_list = jieba.cut(text, cut_all=False) #默认模式分词
    ws = "/ ".join(seg_list)
    for i in ws.split('/'):
        i=i.replace(" ", "")
        if re.match(ur"[\u4E00-\u9FFF]+",i) != None:
            retList.append(i.encode('utf-8'))
    return retList

def rmvStopWord(wordList,stopWords=[word.replace("\r\n","").replace(" ","")  for  word in open('stopwords.txt','r').readlines()]):
    """去停用词
    
    参数：
    wordSet                 词语set集
    stopWords               停用词表
    返回:
    去完停用词后的list
    """
    retList=wordList
    for word in wordList:
        if len(word)<4 or word in stopWords:
            retList.remove(word)
     
    return retList

def TFIDF(term,cls):
    """计算一个term项的TF-IDF值
    
    相乘转化成log相加
    参数:
    term            输入词语（经过分词的词语）
    cls                当前词语所属的文章类别
    """
    cls2Text=cPickle.load(open("model/cls2Text.pkl",'r'))
    #TF
    localNum=2   #除0
    
    