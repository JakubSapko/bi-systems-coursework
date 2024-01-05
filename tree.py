#https://scikit-learn.org/stable/modules/tree.html

from data_store import DataStore
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

class WineTreeClassifier:
    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=5)
        self.data_store = DataStore()
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_store.get_data()

    def train(self):
        self.clf = self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        res = self.clf.predict(self.X_test)
        res_mapped = ["great" if x > 5 else "poor" for x in res]
        return res_mapped

if __name__ == '__main__':
    clf = WineTreeClassifier()
    clf.train()
    print(clf.predict())
    plt.figure()
    plot_tree(clf.clf, filled=True)
    plt.title("Decision Tree")
    plt.savefig("tree.pdf", format="pdf")
    #print(clf.clf.score())
