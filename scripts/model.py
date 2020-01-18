from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle
iris = load_iris()
type(iris)
X = iris.data
Y = iris.target
knn = KNeighborsClassifier(n_neighbors = 11)
knn.fit(X,Y)
pickle.dump(knn, open('model.pickle','wb'))
model = pickle.load(open('model.pickle','rb'))
# print(model.predict([[2, 9, 6]]))