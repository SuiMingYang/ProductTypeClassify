from sklearn.externals import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from collections import Counter

from tkinter import *
top = Tk()
top.title('品类分析工具')
top.geometry("250x150")
# 进入消息循环

text=''
def put_category(l):
    l=loaded_model.predict(vec.transform(jieba.lcut(l)))
    print(l)
    l=sorted(Counter(l).items(),key = lambda kv:(kv[1]),reverse=True)
    for i in l:
        if i[0] !='nan':
            text= i[0]
            L2['text']=text
            print(text)
            break
            


'''
调用模型
'''

filename = './model/extract_secondtype.sav'
loaded_model = joblib.load(filename)

feature_path = './model/feature.pkl'
vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))

L1 = Label(top, text="输入需要分析的商品标题：")
L2 = Label(top)
E1 = Entry(top)
B = Button(top, text ="分析品类",command=lambda: put_category(E1.get()))

L1.pack()
E1.pack()
B.pack()
L2.pack()





top.mainloop()

#  加厚--家纺

# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'2019新款黑色职业短款半身裙')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'女神范套装女春装2019新款时尚休闲慵懒气质御姐洋气阔腿裤两件套')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'【睡眠文胸】无钢圈收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸DQM-4 h405')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'睡眠文胸')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'豪华高端超级无敌被罩四件套')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'全罩杯聚拢大码内衣调整型胸罩运动文胸')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'厚聚拢大码内衣胸罩')))))
# print(put_category(loaded_model.predict(vec.transform(jieba.lcut(u'厚新款修身大码牛仔裤')))))

