#https://scikit-learn.org/stable/modules/tree.html

from sklearn import tree
from ..data.data_store import DataStore

if __name__ == '__main__':
    data_store = DataStore()
    X, y, X_test = data_store.get_data()

    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(X, y)

    clf.predict(X_test)

