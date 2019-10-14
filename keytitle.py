#coding:utf-8
import re
import pandas as pd
import jieba
import jieba.posseg as pseg # 词性标注
from jieba.analyse import ChineseAnalyzer
import numpy as np

'''
分词，统计，计数
'''

class extractkey:
    def __init__(self):
        pass

    def progress(self,proname,rela_dict):
        seg_obj={}
        second_type=[]
        proname=''.join(re.findall(u'[a-zA-Z\d\u4e00-\u9fff]+', proname))
        seg_list=pseg.cut(proname)#,cut_all=False)
        for word in seg_list:
            #if word.flag.find("n")>-1:
            if word.flag=="n":
                try:
                    if rela_dict[word.word]!=None:
                        second_type.append(rela_dict[word.word])
                except Exception:
                    pass

                if word.flag in list(seg_obj.keys()):
                    seg_obj[word.flag].append(word.word)
                else:
                    seg_obj[word.flag]=[]
                    seg_obj[word.flag].append(word.word)
            else:
                pass
        #seg_arr.append(seg_obj)
        #second_arr.append(second_type)
        type_sum={}
        #计数
        for stype in second_type:
            if stype in list(type_sum.keys()):
                type_sum[stype]+=1
            else:
                type_sum[stype]=1
        target_name=""
        target_number=1
        #选择最多的二级品类
        for item in type_sum.keys():
            if type_sum[item]>=target_number:
                target_name=item
                target_number=type_sum[item]
            else:
                pass
        return target_name,second_type#seg_obj


if __name__ == "__main__":
    data=pd.read_excel(u'./data/源数据.xlsx')
    jieba.load_userdict("./dict/expendword.txt")
    '''
    # 分类做键值对字典
    relation=pd.read_csv(u'./dict/relation.csv')
    rela_group=relation.groupby([u'三级品类',u'二级品类'])
    rela_dict={}
    for g in rela_group:
        #print(g[0])
        rela_dict[g[0][0]]=g[0][1]
        #print(g[0][0],g[0][1])
    #print(rela_dict)

    #每个标题，按词性分组
    seg_arr=[]
    second_arr=[]
    target_arr=[]
    extract_obj=extractkey()
    for seg in data[u'产品标题']:
        seg_obj,second_type,target_name=extract_obj.progress(seg,rela_dict)
        seg_arr.append(seg_obj)
        second_arr.append(second_type)
        target_arr.append(target_name)
        #print('精确模式:','/'.join(seg_list))

    #print(seg_arr)
    data[u"三级分词"]=seg_arr
    data[u"二级分词"]=second_arr
    data[u"二级品类结果"]=target_arr
    data.to_csv('./data/品类关联.csv')
    '''

'''
#词频统计
seg_arr=[]
seg_obj={}
for seg in data[u'产品标题']:
    #seg = seg.decode("utf8")
    seg=''.join(re.findall(u'[\d\u4e00-\u9fff]+', seg))
    seg_list=pseg.cut(seg)#,cut_all=False)
    for word in seg_list:
        #print(seg_obj)
        #print(word.word,word.flag)
        if word in list(seg_obj.keys()):
            seg_obj[word]=seg_obj[word]+1
        else:
            seg_obj[word]=1
    #print('精确模式:','/'.join(seg_list))

print(sorted(seg_obj.items(), key=lambda seg:seg[1], reverse=True))
'''

# print(jieba.analyse.extract_tags("2019春季新款韩版大码女装减龄钉珠修身显瘦拼接网纱连衣裙", topK=100, withWeight=False, allowPOS=()))

# seg_list=jieba.cut(sent)
# print('默认模式:','/'.join(seg_list))
# seg_list=jieba.cut_for_search(sent)
# print('搜索引擎模式:','/'.join(seg_list))
# seg_list=jieba.cut(sent,cut_all=True)
# print('全模式:','/'.join(seg_list))

'''
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(['aa','bb','cc'], [1,2,3]).predict(['aa'])
print(y_pred)
'''



