import random
import numpy as np


class KMeans:
    def __init__(self, k=None, n_init=10):
        self.k = k
        self.means = None
        self.n_init = n_init

    def initialize_means_randomly(self, x):
        points = random.sample(range(len(x)), k=self.k)
        self.means = x[points]

    def calculate_distance_to_means(self, a):
        return np.linalg.norm(a-self.means, axis=1)

    @staticmethod
    def calculate_distance(a):
        return np.linalg.norm(a)

    def update_means(self, x):
        n = len(x)
        labels = np.zeros((n,))
        for i in range(n):
            distance = self.calculate_distance_to_means(x[i])
            labels[i] = np.argmin(distance)

        for i in range(self.k):
            clusters = x[[l for l, val in enumerate(labels) if val == i]]
            self.means[i] = np.mean(clusters, axis=0)
        return labels

    def fit(self, x):
        self.initialize_means_randomly(x)
        prev = np.zeros((len(x)))
        while True:
            current_labels = self.update_means(x)
            res = self.calculate_distance(current_labels-prev)
            if res < 10e-5:
                break
            prev = np.copy(current_labels)

    def predict(self, x):
        predictions = []
        for test in x:
            distance = self.calculate_distance_to_means(test)
            predictions.append(np.argmin(distance))
        return predictions

    def score(self, x, y):
        predictions = self.predict(x)
        point = sum(predictions == y) / len(y)
        return point

    def multiple_fit(self, x, y):
        best_means = None
        best_point = 0
        for n in range(self.n_init):
            self.fit(x)
            point = self.score(x, y)
            print(point)
            if point > best_point:
                best_point = point
                best_means = self.means

        self.means = best_means


