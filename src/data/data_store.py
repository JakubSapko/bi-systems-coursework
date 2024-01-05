from ucimlrepo import fetch_ucirepo


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

    def print_data(self):
        print(self.X.head())
        print(self.y.head())
        print(self.X.shape)
