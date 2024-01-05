from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


class DataStore:

    def __init__(self):
        self.set_data()

    def set_data(self):
        # fetch dataset
        wine_quality = fetch_ucirepo(id=186)

        if wine_quality is None:
            print("Failed to fetch data")
            return

        # data (as pandas dataframes)
        self.X = wine_quality.data.features
        self.y = wine_quality.data.targets


        self.build_additional_features()

        # learn, test, validation data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)

    def get_data(self):
        return self.X, self.y, self.X_test

    def build_additional_features(self):
        self.X['total_acidity'] = self.X['fixed_acidity'] + self.X['volatile_acidity']
        self.X['total_sulfur'] = self.X['free_sulfur_dioxide'] + self.X['total_sulfur_dioxide']
        self.X['ratio_sulfur'] = self.X['free_sulfur_dioxide'] / self.X['total_sulfur_dioxide']

    def print_data(self):
        print(self.X.head())
        print(self.y.head())
        print(self.X_test.head())
        print(self.X.shape)
