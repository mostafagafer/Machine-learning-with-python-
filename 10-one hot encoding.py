import pandas as pd
from IPython.display import display

data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education',
                                                                      'education-num', 'marital-status', 'occupation',
                                                                      'relationship', 'race', 'gender', 'capital-gain',
                                                                      'capital-loss', 'hours-per-week', 'native-country',
                                                                      'income'])
data.head()
data = data[['age', 'workclass','education','gender','hours-per-week','occupation','income']]

display(data)

data.columns
data_dummies=pd.get_dummies(data)
data_dummies.columns

features=data_dummies.loc[:,'age':'occupation_ Transport-moving']
X=features.values
y= data_dummies['income_ >50K']


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
logreg.score(X_test,y_test)
