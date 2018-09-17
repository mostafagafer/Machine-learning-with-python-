from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

tree.score(X_train, y_train)  # 1
tree.score(X_test, y_test)  # .937


tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

tree.score(X_train, y_train)  # .98
tree.score(X_test, y_test)  # .95

# pip install graphviz
import graphviz
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='cancer.dot', class_names=[
                'malignant', 'benign'], feature_names=cancer.feature_names, impurity=False, filled=True)


print('Feature importances:{}'.format(tree.feature_importances_))
print(cancer.feature_names)

type(tree.feature_importances_)
n_features = cancer.data.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature importance')
plt.ylabel('Feature ')
plt.show()
