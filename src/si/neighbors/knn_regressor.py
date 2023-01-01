from typing import Callable
import numpy as np
from si.src.si.data.dataset import Dataset
from si.src.si.io.CSV import read_csv
from si.src.si.statistics.euclidean_distance import euclidean_distance
from si.src.si.metrics.rmse import rmse

class KNNRegressor:
    def __init__(self, k: int , distance: Callable = euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        self.dataset = dataset
        return self


    def _get_closest_label(self, sample):
        distances = self.distance(sample, self.dataset.X) # mede a distância passada por parametro(amostra) e as amostras do dataset
        k_nearest_neighbors = np.argsort(distances)[:self.k] # ordena os mais próximos (menor distância)
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors] # obtem os valores em Y
        return np.mean(k_nearest_neighbors_labels) # calcula a média dos valores em Y

    def predict(self, dataset: Dataset) -> np.ndarray:
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X) #aplicar isso a todas as amostras

    def score(self, dataset: Dataset) -> float:
        prediction = self.predict(dataset) # obter previsões
        return rmse(dataset.y, prediction)

if __name__ == '__main__':
    from si.src.si.model_selection.split import train_test_split
    ds = read_csv("C:\\Users\\ruben\\PycharmProjects\\SI\\si\\datasets\\cpu.csv", ",", features = True, label = True)
    ds_train, ds_test = train_test_split(ds, test_size=0.2)
    knn = KNNRegressor(k=4)
    knn.fit(dataset=ds_train)
    s = knn.score(dataset=ds_test)
    print("A precisão do modelo é", s)