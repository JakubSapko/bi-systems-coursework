#https://scikit-learn.org/stable/modules/tree.html

from sklearn import tree

from data_store import DataStore

class TreeWineClassifier:

    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()
        self.data_store = DataStore()
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_store.get_data()

    def train(self):
        self.clf.fit(self.X_train, self.y_train)

    def test(self):
        return self.clf.predict(self.X_test)


if __name__ == '__main__':
    tree_wine_classifier = TreeWineClassifier()
    tree_wine_classifier.train()
    print(tree_wine_classifier.test())

