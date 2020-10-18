import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.01, num_iter = 100, reg='l2', alpha=1, GD='BGD'):
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None
        self.reg = reg
        self.alpha = alpha
        self.GD = GD
        self.m = 100
        self.t0, self.t1 = 5, 50 # learning schedule hyperparameters

    def prepare_X(self, X):
        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate((intercept, X), axis=1)
        return new_X

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def calculate_gradient(self, X, Y):
        gradient = 2 / len(X) * np.dot(X.T, (np.dot(X, self.theta) - Y))
        if self.reg == 'l2':
            gradient += self.alpha * self.theta
        elif self.reg == 'l1':
            gradient += self.alpha * np.sign(self.theta)

        return gradient

    def batch_gradient_decent(self, X, Y):
        for i in range(self.num_iter):
            gradient = self.calculate_gradient(X, Y)
            self.theta -= self.lr * gradient

    def stochastic_gradient_decent(self, X, Y):
        for i in range(self.num_iter):
            for j in range(self.m):
                index = np.random.randint(self.m)
                xj, yj = X[index], Y[index]
                gradient = self.calculate_gradient(xj, yj)
                eta = self.learning_schedule(self.num_iter * self.m + i)
                self.theta -= eta * gradient

    def fit(self, X, Y):
        new_X = self.prepare_X(X)
        self.theta = np.ones((new_X.shape[1]))
        if self.GD == 'BGD':
            self.batch_gradient_decent(new_X, Y)
        elif self.GD == 'SGD':
            self.stochastic_gradient_decent(new_X, Y)

    def predict(self, test_X):
        new_test_X = self.prepare_X(test_X)
        z = np.dot(new_test_X, self.theta)
        return z




