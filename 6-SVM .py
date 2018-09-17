from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
svm=SVC()
svm.fit(X_train, y_train)

svm.score(X_train, y_train)  # 1
svm.score(X_test, y_test)#0.62

plt.plot(X_train.min(axis=0),'o',label="Min")
plt.plot(X_train.max(axis=0),'v',label="Max")
plt.xlabel('Feature index')
plt.ylabel('Feature magnetide in Log scale')
plt.yscale('log')
plt.legend(loc='upper right')

#note that the data is scataared in a logarthmic way


min_train=X_train.min(axis=0)
range_train=(X_train-min_train).max(axis=0)
X_train_scaled=(X_train-min_train)/range_train

X_train_scaled.min(axis=0)
X_train_scaled.max(axis=0)
X_test_scaled=(X_test-min_train)/range_train
svm=SVC()
svm.fit(X_train_scaled,y_train)

svm.score(X_train_scaled, y_train)  # .94
svm.score(X_test_scaled, y_test)#.95


svm=SVC(C=1000)
svm.fit(X_train_scaled,y_train)

svm.score(X_train_scaled, y_train)  # .988
svm.score(X_test_scaled, y_test)#.972



#Uncertinity

svm.decision_function(X_test_scaled)[:20]
svm.decision_function(X_test_scaled)[:20]>0

svm.classes_


#predecting proba

svm=SVC(C=1000,probability=True)
svm.fit(X_train_scaled,y_train)
svm.predict_proba(X_test_scaled[:20])
svm.predict(X_test_scaled)
