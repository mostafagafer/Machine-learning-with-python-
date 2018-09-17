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

#I add the scaller my self

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit(X_train).transform(X_train)
X_test_scaled = scaler.fit(X_test).transform(X_test)

svm=SVC(C=10)
svm.fit(X_train_scaled,y_train)

svm.score(X_train_scaled, y_train)  # .98
svm.score(X_test_scaled, y_test)#.96
