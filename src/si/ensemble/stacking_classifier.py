from typing import List
from si.src.si.data.dataset import Dataset
from si.src.si.metrics.accuracy import accuracy
import numpy as np


class StackingClassifier:
    def __init__(self, models: List[object], final_model: object) -> None:
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> "StackingClassifier":
        for model in self.models:
            model.fit(dataset)  # treina os modelos

        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))  # obtem  as previsões para cada modelo
        self.final_model.fit(Dataset(dataset.X, np.array(predictions).T)) # treina os modelo final
        return self

    def predict(self, dataset: Dataset) -> np.array:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
        y_pred_final = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))
        return y_pred_final

    def score(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)