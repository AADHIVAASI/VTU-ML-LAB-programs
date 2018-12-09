import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df = load_iris()
X = df["data"]
Y = df["target"]

from sklearn.cluster import KMeans
km_cluster = KMeans(n_clusters = 3)
km_cluster.fit(X)
km_predictions = km_cluster.predict(X)
print(km_predictions)
plt.scatter(X[:, 0], X[:, 1], c = km_predictions) # Sepal length vs Sepal width (in cm)
plt.show()

from sklearn.mixture import GaussianMixture
em_cluster = GaussianMixture(n_components = 3)
em_cluster.fit(X)
em_predictions = em_cluster.predict(X)
print(em_predictions)
plt.scatter(X[:, 0], X[:, 1], c = em_predictions)
plt.show()

#Comparing their accuracies
from sklearn.metrics import accuracy_score, confusion_matrix
km_accuracy = accuracy_score(Y, km_predictions)
em_accuracy = accuracy_score(Y, em_predictions)
km_confusion = confusion_matrix(Y, km_predictions)
em_confusion = confusion_matrix(Y, em_predictions)
print("Accuracy of KMeans is ",km_accuracy)
print("Accuracy of EM is ",em_accuracy)
print("Confusion matrix of KMeans: \n", km_confusion)
print("Confusion matrix of EM: \n", em_confusion)
