from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

iris = load_iris()


print(iris.data)

print(iris.feature_names)  # act as coulmn index
print(iris.target)
print(iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

gbrt=GradientBoostingClassifier(learning_rate=0.01, random_state=0)
gbrt.fit(x_train,y_train)
gbrt.decision_function(x_test[:10])
gbrt.predict_proba(x_test[:10]) # highe number refer to the class
