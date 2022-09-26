import numpy as np
import pandas as pd

class Dataset:
    def __init__(self, x y, features, label):
        self.x = x
        self.y = y
        self.features = features
        self.label = label

    def get_shape():
        return self.y.shape

    def has_label():
        if self.y is None:
            return False
        else: return True


    def get_classes(self):
        if self.y is None:
            return
        return np.unique(self.y)

    def get_mean(self):
        return np.mean(self.x, axis=0)

    def summary(self):
        return pd.DataFrame(
            {'mean': self.get_mean(),
            'median': self.get_median(),
        })



