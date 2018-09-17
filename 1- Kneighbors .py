from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

cancer = load_breast_cancer()
print(cancer.DESCR)


cancer.feature_names
cancer.feature
cancer.target_names
cancer.target

raw_data = pd.read_csv('C:/Users/Mostafa_2/Downloads/data.csv', delimiter=',')
raw_data.tail()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_train, y_train)  # .94
knn.score(X_test, y_test)  # .93

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
neighbors_setting = range(1, 11)
for n_neighbor in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=n_neighbor)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='Accuracy of the training set')
plt.plot(neighbors_setting, test_accuracy, label='Accuracy of the test set')

# optimum accuracy at knn=7
