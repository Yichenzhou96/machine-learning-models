from DecisionTree import *
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = DecisionTree()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('prediction score: {}'.format(sum(predictions == y_test)/len(y_test)))
model.print_tree()