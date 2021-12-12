import numpy as np
import matplotlib.pyplot as plt
from numpy import array, int64

from sklearn.datasets import make_blobs
# ---------------------------250 образцов, 4 класса, хорошо разделимы-----------------------
"""
X, y = make_blobs(n_samples=250, n_features=5, centers=4, random_state=1, cluster_std=0.3)
print("------")
print(X)
print(y)

print(X[1:5, 1])
print(y[1:5])

y[:10]

np.unique(y, return_counts=True)

(array([0, 1, 2]), array([34, 33, 33], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()
"""
# --------------------------250 образцов, 5 классов, хорошо разделимы-------------------------
"""
X2, y2 = make_blobs(n_samples=250, n_features=3, centers=5, random_state=1, cluster_std=0.3)
print(X2.shape)
print(y2.shape)
print(X2[1:5, 1])
print(y2[:10])

np.unique(y2, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X2[:, 0], X2[:, 1], c=y2)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()
"""
# --------------------------250 образцов, 4 класса, 2 из них плохо разделены------------
"""
X3, y3 = make_blobs(n_samples=250, n_features=5, centers=4, random_state=1, cluster_std=0.3)
print(X3.shape)
print(y3.shape)
print(X3[1:5, 1])
print(y3[:10])

np.unique(y3, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X3[:, 0], X3[:, 1], c=y3)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()
"""
# --------------------------29 образцов, 2 класса, меняется генерация чисел------------
"""
X5, y5 = make_blobs(n_samples=29, n_features=3, centers=2, random_state=5, cluster_std=0.3)
print(X5.shape)
print(y5.shape)
print(X5[1:5, 1])
print(y5[:10])

np.unique(y5, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X5[:, 0], X5[:, 1], c=y5)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()

X6, y6 = make_blobs(n_samples=29, n_features=3, centers=2, random_state=3, cluster_std=0.3)
print(X6.shape)
print(y6.shape)
print(X6[1:5, 1])
print(y6[:10])

np.unique(y6, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X6[:, 0], X6[:, 1], c=y6)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()

X7, y7 = make_blobs(n_samples=29, n_features=3, centers=2, random_state=1, cluster_std=0.3)
print(X7.shape)
print(y7.shape)
print(X7[1:5, 1])
print(y7[:10])

np.unique(y7, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X7[:, 0], X7[:, 1], c=y7)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()

"""

centers = [[1,1,1],[1,3,1],[1,5,1],[1,7,1]]
X8, y8 = make_blobs(n_samples=157, n_features=2, centers=centers, random_state=1, cluster_std=0.1)
print(X8)
print(y8)
print(X8[1:5, 1])
print(y8[:10])

np.unique(y8, return_counts=True)

(array([0, 1]), array([50, 50], dtype=int64))

plt.figure(figsize=(8, 8))
plt.scatter(X8[:, 0], X8[:, 1], c=y8)
plt.xlabel("First")
plt.ylabel("Second")
plt.show()
