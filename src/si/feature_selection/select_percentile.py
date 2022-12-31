from typing import Callable

import numpy as np
import pandas as pd

from si.src.si.data.dataset import Dataset
from si.src.si.statistics.f_classification import f_classification


class SelectPercentile:
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.25):

        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':

        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        n_amostras = len(list(dataset.features)) # guarda na variavel o número total de amostras
        amostras_ate_ao_percentil = int(n_amostras * self.percentile) # multiplica pelo percentil
        idxs = np.argsort(self.F)[-amostras_ate_ao_percentil:] # seleciona as ultimas amostras de acordo com o percentil
        features = np.array(dataset.features)[idxs] # converte para um array
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)


'''
if __name__ == '__main__':
    from si.src.si.data.dataset import Dataset

    dataframe = pd.read_csv("C:\\Users\\ruben\\PycharmProjects\\SI\\si\\datasets\\iris.csv")
    dataset = Dataset.from_dataframe(dataframe, label='class') # chamar o from_dataframe
    dataset = Dataset(X=np.array([[0, 1, 2],
                                   [2, 1, 0],
                                   [1, 2, 0]]),
                       y=np.array([0, 1, 0]),
                       features=["A", "B", "C"],
                      label="y")

    selector = SelectPercentile(percentile=1/3)
    selector = selector.fit_transform(dataset=dataset)
    print('Olá')
    print(selector.features)
    print('Adeus')

'''