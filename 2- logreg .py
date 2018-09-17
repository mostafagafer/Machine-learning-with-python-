from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg.score(X_train, y_train)  # .955
log_reg.score(X_test, y_test)  # .958

log_reg100 = LogisticRegression(C=100)
log_reg100.fit(X_train, y_train)
log_reg100.score(X_train, y_train)  # .97
log_reg100.score(X_test, y_test)  # .965


log_reg001 = LogisticRegression(C=0.00100)
log_reg001.fit(X_train, y_train)
log_reg001.score(X_train, y_train)  # .922
log_reg001.score(X_test, y_test)  # .937
