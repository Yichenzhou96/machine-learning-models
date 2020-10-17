import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, num_iter = 100):
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None

    def prepare_X(self, X):
        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate((intercept, X), axis=1)
        return new_X

    def fit(self, X, Y):
        new_X = self.prepare_X(X)
        self.theta = np.ones((new_X.shape[1]))

        # Batch gradient decent
        for i in range(self.num_iter):
            gradient = 2/len(new_X) * np.dot(new_X.T, (np.dot(new_X, self.theta) - Y))
            self.theta -= self.lr * gradient

    def predict(self, test_X):
        new_test_X = self.prepare_X(test_X)
        z = np.dot(new_test_X, self.theta)
        return z




