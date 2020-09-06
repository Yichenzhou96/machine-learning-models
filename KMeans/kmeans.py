import random
import numpy as np
class kmeans:
    def __init__(self, k=None):
        self.k = k
        self.means = None

    def initialize_means_randomly(self, x):
        points = random.sample(range(len(x)), k=self.k)
        self.means = x[points]

    def calculate_distance(self, a):
        return np.linalg.norm(a-self.means, axis=1)

    def update_means(self, x):
        labels = []
        for instance in x:
            distance = self.calculate_distance(instance)
            labels.append(np.argmin(distance))

        for i in range(self.k):
            clusters = x[[l for l, val in enumerate(labels) if val == i]]
            self.means[i] = np.mean(clusters, axis=0)

        return labels

    def fit(self, x):
        self.initialize_means_randomly(x)
        while True:
            prev = []
            current_labels = self.update_means(x)
            if np.linalg.norm([a - b for a, b in zip(prev, current_labels)]) < 10e-5:
                break

    def predict(self, x):
        pass


from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = kmeans(3)
model.fit(X_train)

print(model.means)


