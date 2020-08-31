import numpy as np

class SMO:
    def __init__(self, C=1, tol=0.01, max_passes=10, kernel_option='RBF', sigma=1, degree=3):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel_option = kernel_option
        self.sigma = sigma
        self.degree = degree
        self.weights = None
        self.b = None

    def calculate_kernel(self, xi, xj):
        inner_product = 0
        if self.kernel_option == 'RBF':
            diff = np.abs(xi - xj)
            inner_product = np.exp(np.dot(diff, diff) / (-2 * self.sigma**2))
        elif self.kernel_option == 'linear':
            inner_product = np.dot(xi, xj)
        elif self.kernel_option == 'poly':
            inner_product = (np.dot(xi, xj) + 1) ** self.degree
        else:
            raise Exception("Sorry, not valid kernel function")

        return inner_product

    def kernel_matrix(self, xi, X):
        kernel = list(map(lambda x: self.calculate_kernel(xi, x), X))
        return kernel

    def calculate_error(self, xi, yi, X, Y, alpha, b):
        kernel = self.kernel_matrix(xi, X)
        pred = np.sign(np.dot(np.multiply(alpha, Y), kernel) + b)
        error = pred - yi
        return error

    def calculate_eta(self, xi, xj):
        return 2 * self.calculate_kernel(xi, xj) - self.calculate_kernel(xi, xi) - self.calculate_kernel(xj, xj)

    def update_weight(self, alpha, X, Y):
        self.weights = np.dot(np.multiply(alpha, Y), X)

    def fit(self, train_x, train_y):
        m, n = train_x.shape
        alpha = np.zeros((m, ))
        self.b = 0
        self.weights = np.zeros((n, ))
        passes = 0

        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(m):
                errori = self.calculate_error(train_x[i], train_y[i], train_x, train_y, alpha, self.b)
                if (train_y[i] * errori < -self.tol and alpha[i] < self.C) or \
                        (train_y[i] * errori > self.tol and alpha[i] > 0):

                    j = np.random.choice([x for x in range(m) if x != i])
                    errorj = self.calculate_error(train_x[j], train_y[j], train_x, train_y, alpha, self.b)

                    if train_y[i] != train_y[j]:
                        L = np.maximum(alpha[j] - alpha[i], 0)
                        H = np.minimum(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = np.maximum(0, alpha[i] + alpha[j] - self.C)
                        H = np.minimum(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = self.calculate_eta(train_x[i], train_x[j])
                    if eta >= 0:
                        continue

                    old_alphai, old_alphaj = alpha[i], alpha[j]
                    new_alphaj = old_alphaj - train_y[j] * (errori - errorj) / eta

                    if new_alphaj > H:
                        alpha[j] = H
                    elif new_alphaj < L:
                        alpha[j] = L
                    else:
                        alpha[j] = new_alphaj

                    if np.abs(alpha[j] - old_alphaj) < 10e-5:
                        continue

                    alpha[i] = old_alphai + train_y[i] * train_y[j] * (old_alphaj - alpha[j])

                    b1 = self.b - errori - train_y[i] * (alpha[i] - old_alphai) * self.calculate_kernel(train_x[i], train_x[i])\
                         - train_y[j] * (alpha[j] - old_alphaj) * self.calculate_kernel(train_x[i], train_x[j])

                    b2 = self.b - errori - train_y[i] * (alpha[i] - old_alphai) * self.calculate_kernel(train_x[i], train_x[j])\
                         - train_y[j] * (alpha[j] - old_alphaj) * self.calculate_kernel(train_x[j], train_x[j])

                    if 0 < alpha[i] < self.C:
                        self.b = b1
                    elif 0 < alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

        self.update_weight(alpha, train_x, train_y)

    def calculate_pred(self, xi):
        return np.sign(np.dot(self.weights, xi) + self.b)

    def predict(self, test_x):
        m, n = test_x.shape
        prediction = np.zeros((m, ))
        for i in range(m):
            prediction[i] = self.calculate_pred(test_x[i])

        return prediction

    def score(self, test_x, test_y):
        m, n = test_x.shape
        count = 0
        for i in range(m):
            prediction = self.calculate_pred(test_x[i])
            if prediction == test_y[i]:
                count += 1

        return count / m



