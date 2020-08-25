import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.01, num_iter=10000):
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None
        
    def prepare_X(self, X):
        intercept = np.ones((X.shape[0], 1))
        new_X = np.concatenate((intercept, X), axis=1)
        return new_X
    
    def logistic(self, z):
        return 1/(1 + np.exp(-z))
    
    def compute_cost(self, h, y):
        epsilon = 1e-5
        return -y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)
    
    def fit(self, X, y):
        new_X = self.prepare_X(X)
        self.theta = np.zeros((X.shape[1]))
        
        for i in range(self.num_iter):
            z = np.dot(new_X, self.theta)
            h = self.logistic(z)
            
            gradient = np.dot((h- y), new_X)/X.shape[0]
            self.theta -= self.lr * gradient
            
            cost = self.compute_cost(h, y)
#             print(cost)
        
    def predict(self, test_X):
        new_test_X = self.prepare_X(test_X)
        z = np.dot(new_test_X, self.theta)
        h = self.logistic(z)
        return h
