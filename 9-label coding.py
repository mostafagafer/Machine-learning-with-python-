from sklearn import preprocessing

labels = ['setosa', 'versicolor', 'virginica']

encoder = preprocessing.LabelEncoder()
encoder.fit(labels)

for i, item in enumerate(encoder.classes_):
    print(item, '=>', i)

more_labels = ['versicolor', 'versicolor', 'virginica', 'setosa', 'versicolor']
more_labels_encoded = encoder.transform(more_labels)

more_labels
more_labels_encoded
