import numpy as np
import sys
import os


from DecisionTree import DecisionTree

class AdaBoost:
    def __init__(self, n=50, eta=1):
        self.n_predictor = n
        self.eta = eta
        self.alphas = [1 for _ in range(self.n_predictor)]
        self.predictors = [None for _ in range(self.n_predictor)]

    def compute_predictor_weight(self, y_hat, y, w):
        weight_error = 0
        total_weight = 0
        for y_hat_i, y_i, w_i in zip(y_hat, y, w):
            if y_hat_i != y_i:
                weight_error += w_i

            total_weight += w_i

        weight_error_rate = weight_error / total_weight if total_weight != 0 else 0
        predictor_weight = self.eta * np.log((1-weight_error_rate) / weight_error_rate)
        return predictor_weight

    def update_instance_weight(self, y_hat, y, w, alpha):
        size = len(w)
        for i in range(size):
            if y_hat[i] != y[i]:
                w[i] = w[i] * np.exp(alpha)

        w = w/sum(w)

        return w

    def fit(self, X, y):
        size = len(X)
        weights = [1/size for _ in range(size)]
        for i in range(self.n_predictor):
            X_ = np.dot(X, weights)
            predictor = DecisionTree()
            predictor.fit(X_, y)
            y_hat = predictor.predict(X)
            predictor_weight = self.compute_predictor_weight(y_hat, y, weights)
            weights = self.update_instance_weight(y_hat, y, weights, predictor_weight)
            self.alphas[i] = predictor_weight
            self.predictors[i] = predictor

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

