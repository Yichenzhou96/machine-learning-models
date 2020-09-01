from MaximumDecisionTree import MaximumDecisionTree

class RandomForest:
    def __init__(self, n_estimator=5, maximum_samples=100, sampling='bagging'):
        self.n_estimator = n_estimator
        self.sampling = sampling
        self.maximum_sample = maximum_samples

    def sampling(self, x, y):
        pass

    def fit(self, x, y):
        pass