from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics


df = pd.read_csv('C:/Users/Mostafa_2/Downloads/data.csv', delimiter=',')
df.tail()
df.dtypes
df.fillna(-99999, inplace=True)

X = np.array(df.drop(['diagnosis'], 1))
y = np.array(df['diagnosis'])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)  # .84
knn.score(X_test, y_test)  # .73


K_range = range(1, 26)
K_range
score = []
for k in K_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score.append(metrics.accuracy_score(y_test, y_pred))
print(score)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# extentiate our KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


training_accuracy = []
test_accuracy = []
neighbors_setting = range(1, 20)
for n_neighbor in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting, test_accuracy, label='Accuracy of the test set')
