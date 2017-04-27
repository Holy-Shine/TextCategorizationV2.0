# coding=utf-8
import re
import jieba
import cPickle
import chardet

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
        i=i.replace(" ",'')
        if re.match(ur"[\u4E00-\u9FFF]+",i) != None:
            retList.append(i.encode('UTF-8'))
    print len(retList)
    return retList

def rmvStopWord(wordSet,stopWords):
    """去停用词
    
    参数：
    wordSet-词语set集
    返回:
    去完停用词后的list
    """
    retList=list(wordSet)
    for word in wordSet:
        if len(word)<4 or word in stopWords:
            retList.remove(word)
     
    return retList

 
# file=open('train.txt','r')
# dic={}
#    
# for line in file:
#     tp=line.split(' ')
#     if tp[0] not in dic:
#         dic[tp[0]]=[]
#     dic[tp[0]].extend(segText(tp[1]))
#           
# for key in dic.keys():
#     dic[key]=list(set(dic[key]))
#           
# cPickle.dump(dic, open("dict.pkl",'w'), protocol=0)
dic=cPickle.load(open("dict.pkl",'r'))
stopWords=[word.replace("\r\n","").replace(" ","")  for  word in open('stopwords.txt','r').readlines()]
for key in dic.keys():
    print key,len(dic[key])
    dic[key]=rmvStopWord(dic[key], stopWords)
    print key,len(dic[key])


    