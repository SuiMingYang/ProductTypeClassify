# ProductTypeClassify
商品文本标题分类

---

### 一、项目目标
从商品标题(productTitle)中分类出商品品类(secondtype)。

---

### 二、环境配置
脚本：python3.6,anaconda3
安装包：jieba分词，scikit-learn

---

### 三、数据准备
数据是用爬虫抓的某网站的商品标题。
分类标签自定义为：
`['上衣','其它','包包','外套','套装','家纺','日常','日用','裙子','裤子','连衣裙','配饰','链表','鞋']`
对商品标题人工标记分类做训练数据。

---

### 四、数据处理
分词，去除停用词，转成词向量
```
import jieba
import pandas as pd
import random

sentences=[]
df = pd.read_csv('./data/product.csv',encoding='UTF-8')
for i,item in enumerate(df['productTitle']):
    segs=jieba.lcut(item)
    sentences.append((" ".join(segs),df['secondtype'][i]))
```

---

### 五、模型训练
```
random.shuffle(sentences)

from sklearn.model_selection import train_test_split

x,y = zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234)

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    analyzer='word',
    ngram_range=(1,4),
    max_features=22000
)
vec.fit(x_train)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score
import numpy as np

#使用svm分析
from sklearn.svm import SVC
svm = SVC(kernel='linear')#快
#SVC()#默认rbf核，慢
svm.fit(vec.transform(x_train),y_train)
print(svm.score(vec.transform(x_test),y_test))

#写入字典或者离线数据文件
from sklearn.externals import joblib
import pickle
#保存模型
filename = 'extract_secondtype.sav'
joblib.dump(svm, filename)
```
准确率：
```
->
88.95%
```

---

### 六、模型检验
```
# load the model from disk
svm = joblib.load(filename)
result = svm.score(vec.transform(x_test),y_test)
#print(result)

print(svm.score(vec.transform(x_test),y_test))
print(svm.predict(vec.transform(jieba.lcut(u'2019新款黑色职业短款半身裙'))))
print(svm.predict(vec.transform(jieba.lcut(u'女神范套装女春装2019新款时尚休闲慵懒气质御姐洋气阔腿裤两件套'))))
print(svm.predict(vec.transform(jieba.lcut(u'【睡眠文胸】新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸DQM-4 h405'))))
print(svm.predict(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣调整型胸罩运动文胸'))))
print(svm.predict(vec.transform(jieba.lcut(u'新款无钢圈薄款收副乳全罩杯加厚聚拢大码内衣胸罩'))))

```
结果输出：
```
->
['nan' 'nan' 'nan' 'nan' 'nan' '裙子']

['nan' 'nan' '套装' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' '裤子' '套装']

['nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' '日用' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan']

['nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' '日用' 'nan' 'nan' 'nan' 'nan']

['nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' 'nan' '日用'
 'nan']
```

结果中可以看到，每个标题被分词，转换成词向量，并建立单个词和类型的对应权重，返回的结果数组中出现最多的词即为该标题的类型。

---

完整代码地址：[https://github.com/SuiMingYang/ProductTypeClassify](https://github.com/SuiMingYang/ProductTypeClassify)

