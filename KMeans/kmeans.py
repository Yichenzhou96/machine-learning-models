import random
import numpy as np


class kmeans:
    def __init__(self, k=None):
        self.k = k
        self.means = None

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
            print(res)
            if res < 10e-5:
                break
            prev = np.copy(current_labels)

    def predict(self, x):
        predictions = []
        for test in x:
            distance = self.calculate_distance_to_means(test)
            predictions.append(np.argmin(distance))
        return predictions


from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = kmeans(3)
model.fit(X_train)
predictions = model.predict(X_test)
print('prediction score: {}'.format(sum(predictions == y_test)/len(y_test)))


