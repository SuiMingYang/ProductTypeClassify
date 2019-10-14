import jieba
import pandas as pd
from keytitle import extractkey
import random

"""
使用词库对商品库标注，生成训练集
"""

jieba.load_userdict("./data/expendword.txt")
data=pd.read_csv('./data/product_2019-09-20.csv')

def relation_dict():
    #分词加载，预处理字典
    #data=pd.read_excel(u'./data/源数据.xlsx')

    # 分类做键值对字典
    relation=pd.read_csv(u'./data/relation.csv')
    rela_group=relation.groupby([u'二级类目',u'三级类目'])
    rela_dict={}
    for g in rela_group:
        #print(g[0])
        rela_dict[g[0][1]]=g[0][0]

    return rela_dict

rela_dict=relation_dict()

target=[]
#obj=[]
second=[]
extract_obj=extractkey()
for i,item in enumerate(data['product_title']):
    target_name,second_obj=extract_obj.progress(item,rela_dict)
    target.append(target_name)
    #obj.append(obj)
    second.append(second_obj)

print(len(second))
new_data={
    'id':data['product_id'],
    'title':data['product_title'],
    'category':target,
    #'obj':obj,
    'list':second
}
new=pd.DataFrame(new_data,index=None)
new.to_csv('./data/qingsheceshi.csv')


