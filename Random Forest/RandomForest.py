from MaximumDecisionTree import MaximumDecisionTree
import random
import numpy as np

class RandomForest:
    def __init__(self, n_estimator=5, maximum_samples=100, max_depth = 2, sampling_type='bagging'):
        self.n_estimator = n_estimator
        self.sampling_type = sampling_type
        self.maximum_sample = maximum_samples
        self.max_depth = max_depth

    def sampling(self, x, y):
        subset = random.sample(range(len(x)), k=self.maximum_sample)
        return x[subset], y[subset]

    def fit_single_tree(self, x, y):
        decision_tree = MaximumDecisionTree(max_depth=self.max_depth)
        decision_tree.fit(x, y)
        # print(decision_tree.print_tree())
        return decision_tree

    def fit(self, x, y):
        self.random_forest = []
        for n in range(self.n_estimator):
            sub_x, sub_y = self.sampling(x, y)
            self.random_forest.append(self.fit_single_tree(sub_x, sub_y))

    def predict(self, test):
        n_test = len(test)
        total_predictions = np.zeros((self.n_estimator, n_test))
        predictions = []
        for n in range(self.n_estimator):
            decision_tree = self.random_forest[n]
            total_predictions[n] = decision_tree.predict(test)

        for i in range(n_test):
            unique, counts = np.unique(total_predictions[:, i], return_counts=True)
            predictions.append(unique[np.argmax(counts)])

        return predictions


from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForest()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print('prediction score: {}'.format(sum(predictions == y_test)/len(y_test)))