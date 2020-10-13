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
