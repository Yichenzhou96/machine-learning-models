import numpy as np
from random import choices
from decisionTree.DecisionTree import DecisionTree


class AdaBoost:
    def __init__(self, n=2, eta=1):
        self.n_predictor = n
        self.eta = eta
        self.alphas = [1 for _ in range(self.n_predictor)]
        self.predictors = []

    def compute_predictor_weight(self, y_hat, y, w):
        weight_error = 0
        total_weight = 0
        for y_hat_i, y_i, w_i in zip(y_hat, y, w):
            if y_hat_i != y_i:
                weight_error += w_i

            total_weight += w_i

        if total_weight == 0:
            weight_error_rate = 0
        else:
            weight_error_rate = weight_error / total_weight

        if weight_error_rate == 0:
            return 0
        else:
            return self.eta * np.log((1-weight_error_rate) / weight_error_rate)

    def update_instance_weight(self, y_hat, y, w, alpha):
        size = len(w)
        for i in range(size):
            if y_hat[i] != y[i]:
                w[i] = w[i] * np.exp(alpha)

        w = w/sum(w)

        return w

    def create_new_training_dataset(self, X, w, y):
        size = X.shape
        X_new = np.zeros(size)
        y_new = np.zeros(len(y))
        population = [n for n in range(len(X))]
        for i in range(size[0]):
            # print(choices(X, w)[0])
            index = choices(population, w)[0]
            X_new[i] = X[index]
            y_new[i] = y[index]

        return X_new, y_new

    def fit(self, X, y):
        size = len(X)
        weights = np.zeros((size,)) + 1/size

        for i in range(self.n_predictor):
            X, y = self.create_new_training_dataset(X, weights, y)
            predictor = DecisionTree()
            predictor.fit(X, y)
            y_hat = predictor.predict(X)
            predictor_weight = self.compute_predictor_weight(y_hat, y, weights)
            weights = self.update_instance_weight(y_hat, y, weights, predictor_weight)
            self.alphas[i] = predictor_weight
            self.predictors.append(predictor)

    def predict(self, test):
        pre = []
        for instance in test:
            y_ = [0 for _ in range(self.n_predictor)]
            classes = {}
            for j in range(self.n_predictor):
                y_[j] = self.predictors[j].predict(instance)
                if y_[j] not in classes:
                    classes[y_[j]] = self.alphas[j]
                else:
                    classes[y_[j]] += self.alphas[j]

            pre.append(max(classes, key=classes.get))

        return pre

