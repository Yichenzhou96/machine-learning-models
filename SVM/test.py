from SMO import *
from sklearn.model_selection import train_test_split
import numpy as np

x = np.random.normal(size=(5,))
y = np.outer(x, x)
z = np.random.multivariate_normal(np.zeros(5)-2, y, size=100)

x1 = np.random.normal(size=(5,))
y1 = np.outer(x1, x1)
z1 = np.random.multivariate_normal(np.zeros(5)+2, y1, size=100)

X_train = np.r_[z, z1]

Y_train = np.r_[[1 for x in range(100)], [-1 for x in range(100)]]

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

model = SMO()
model.fit(X_train, y_train)
print('score:', model.score(X_test, y_test))