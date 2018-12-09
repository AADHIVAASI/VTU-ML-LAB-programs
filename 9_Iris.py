from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset=load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))




from sklearn.datasets import load_iris

df = load_iris()
data = df["data"]
target = df["target"]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_test)
print("Predictions: \n", knn_predictions)

from sklearn.metrics import accuracy_score, confusion_matrix
train_accuracy = accuracy_score(Y_train, knn.predict(X_train))
test_accuracy = accuracy_score(Y_test, knn.predict(X_test))
train_confusion = confusion_matrix(Y_train, knn.predict(X_train))
test_confusion = confusion_matrix(Y_test, knn.predict(X_test))
print("Training accuracy: ", train_accuracy)
print("Testing accuracy: ", test_accuracy)
print("Training confusion matrix: \n", train_confusion)
print("Testing confusion matrix: \n", test_confusion)
