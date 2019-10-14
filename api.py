from sklearn.externals import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import jieba
from collections import Counter

"""
调用代码
"""

if __name__ == "__main__":
    def put_category(l):
        l=loaded_model.predict(vec.transform(jieba.lcut(l)))
        l=sorted(Counter(l).items(),key = lambda kv:(kv[1]),reverse=True)
        for i in l:
            if i[0] !='nan':
                text = i[0]
                print(text)
                return text
        return ""

    modelname = './model/extract_secondtype.sav'
    loaded_model = joblib.load(modelname)

    featurename = './model/feature.pkl'
    vec = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(featurename, "rb")))

    put_category('大码宽松长袖衬衣')
    