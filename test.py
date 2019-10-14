import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = SVC(probability=True) # LogisticRegression(solver='lbfgs', multi_class='multinomial',
#                           random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
eclf1 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
eclf1 = eclf1.fit(X, y)
print('1',eclf1.predict(X))

np.array_equal(eclf1.named_estimators_.lr.predict(X),
               eclf1.named_estimators_['lr'].predict(X))

eclf2 = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
        voting='soft')
eclf2 = eclf2.fit(X, y)
print('2',eclf2.predict(X))

eclf3 = VotingClassifier(estimators=[
       ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
       voting='soft', weights=[2,1,1],
       flatten_transform=True)
eclf3 = eclf3.fit(X, y)
print('3',eclf3.predict(X))

print(eclf3.transform(X).shape)



lr = LogisticRegression(n_jobs=-1, C=8)  # meta classifier
from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr, use_probas=True, verbose=3)

sclf.fit(X, y)
print('4',eclf3.predict(X))

