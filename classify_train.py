import jieba
import pandas as pd
import random

'''
训练模型，保存
'''
jieba.load_userdict("./data/expendword.txt")
sentences=[]
df = pd.read_csv('./data/qingsheceshi.csv',encoding='UTF-8')
df = df
for i,item in enumerate(df['title']):
    segs=jieba.lcut(item)
    sentences.append((" ".join(segs),df['category'][i]))

random.shuffle(sentences)

from sklearn.model_selection import train_test_split

x,y = zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    max_features=25000
)
vec.fit(x)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score
import numpy as np


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = SVC(gamma='scale', kernel='rbf', probability=True)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
eclf1.fit(vec.transform(x_train).toarray(),y_train)
print(eclf1.score(np.array(vec.transform(x_test)),y_test))
print(eclf1.predict(vec.transform(jieba.lcut(u'2019新款黑色职业短款半身裙')).toarray()))
print(eclf1.predict_proba(vec.transform(jieba.lcut(u'女神范套装女春装2019新款时尚休闲慵懒气质御姐洋气阔腿裤两件套'))))
print(eclf1.predict_proba(vec.transform(jieba.lcut(u'【睡眠文胸】新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸DQM-4 h405'))))
print(eclf1.predict_proba(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸'))))
print(eclf1.predict_proba(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣胸罩'))))
print(eclf1.predict_proba(vec.transform(jieba.lcut(u'女王新款修身大码牛仔裤'))))


# #使用svm分析
# from sklearn.svm import SVC
# svm = SVC(kernel='linear')#快
# # SVC()#默认rbf核，慢
# svm.fit(vec.transform(x_train),y_train)
# print(svm.score(vec.transform(x_test),y_test))
# print(svm.predict(vec.transform(jieba.lcut(u'2019新款黑色职业短款半身裙'))))
# print(svm.predict(vec.transform(jieba.lcut(u'女神范套装女春装2019新款时尚休闲慵懒气质御姐洋气阔腿裤两件套'))))
# print(svm.predict(vec.transform(jieba.lcut(u'【睡眠文胸】新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸DQM-4 h405'))))
# print(svm.predict(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸'))))
# print(svm.predict(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣胸罩'))))
# print(svm.predict(vec.transform(jieba.lcut(u'女王新款修身大码牛仔裤'))))


# #写入字典或者离线数据文件
# from sklearn.externals import joblib
# import pickle
# #保存模型
# filename = './model/extract_secondtype.sav'
# joblib.dump(svm, filename)
# # load the model from disk

# feature_path = './model/feature.pkl'
# with open(feature_path, 'wb') as fw:
#     pickle.dump(vec.vocabulary_, fw)


