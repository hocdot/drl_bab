import numpy as np

class StandardScaler:

    def __init__(self, mean=0, std=0):
        self.mean = mean
        self.std = std
        self.num_data = 0
        self.sum_square = 0
        self.sum = 0
        self.fitted = False

    def fit(self, data):
        new_num_data = self.num_data + np.size(data)

        new_mean = (self.mean * self.num_data + np.sum(data)) / new_num_data

        new_std = np.sqrt((new_num_data * (self.sum_square + np.sum(data ** 2)) - (self.sum + np.sum(data)) ** 2) /
                       (new_num_data ** 2))

        self.num_data = new_num_data
        self.mean = new_mean
        self.std = new_std
        self.sum_square = self.sum_square + np.sum(data ** 2)
        self.sum = self.sum + np.sum(data)
        self.fitted = True

    def transform(self, data):
        _data = data.clone()
        _data = (_data - self.mean) / self.std
        return _data

    def fit_transform(self, data):
        _data = data.detach().cpu().numpy()
        self.fit(_data)
        return self.transform(data)
    
    def __repr__(self):
        return f'StandardScaler(mean={self.mean}, std={self.std}'