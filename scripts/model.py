# load iris dataset from sklearn built in datasets
from sklearn.datasets import load_iris
#import KNeighboursClassifier from sklearn neighbours lib
from sklearn.neighbors import KNeighborsClassifier
#pickle used for reading a pre-computed ml model.
import pickle
#load iris
iris = load_iris()
#initiliaze x as data of iris dataset
X = iris.data
#initiliaze y as target variables of iris dataset
Y = iris.target
#knn is now a classifier object that has been initialized
#most optimal value is n_neighbors = 10-17 with an accuracy of approx 98.5%
knn = KNeighborsClassifier(n_neighbors = 11)
#fit the classifier model 
knn.fit(X,Y)
#dump knn model in model.pickle with write-binary mode
pickle.dump(knn, open('model.pickle','wb'))
#create a classifier model 'model' that can be used for prediction
model = pickle.load(open('model.pickle','rb'))
#model can be tested by using predict function as depicted below
# print(model.predict([[2, 9, 6]]))
