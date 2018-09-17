import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
rng=np.random.RandomState(42)
noise= rng.normal(size=(len(cancer.data),50))

X_w_noise=np.hstack([cancer.data,noise])
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)


select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print('X_train.shape is: {}'.format(X_train.shape))
print('X_train_selected.shape is: {}'.format(X_train_selected.shape))

mask=select.get_support()

mask

plt.matshow(mask.reshape(1,-1),cmap="gray_r")


X_test_selected = select.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
logreg.score(X_test, y_test)

logreg.fit(X_train_selected, y_train)
logreg.score(X_test_selected, y_test)



#Model based feature model_selection

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select=SelectFromModel(RandomForestClassifier(n_estimators=100,random_state=42),threshold='median')
select.fit(X_train,y_train)
X_train_s=select.transform(X_train)
X_train.shape
X_train_s.shape

mask=select.get_support()
plt.matshow(mask.reshape(1,-1),cmap="gray_r")
plt.xlabel("index of the features")

X_test_s=select.transform(X_test)
score=LogisticRegression().fit(X_train_s,y_train).score(X_test_s,y_test)
score#0.9508771929824561
