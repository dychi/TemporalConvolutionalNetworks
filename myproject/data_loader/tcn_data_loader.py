from base.base_data_loader import BaseDataLoader

class DataGenerator(BaseDataLoader):
    def __init__(self, config):
        super(DataGenerator, self).__init__(config)
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.data()


    def get_train_data(self):
        return self.X_train, self.y_train


    def get_test_data(self):
        return self.X_test, self.y_test


