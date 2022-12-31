import numpy as np
from si.src.si.data.dataset import Dataset
from si.src.si.metrics.accuracy import accuracy


class StackingClassifier:

    def __init__(self, models: list, final_model: Callable):
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> "StackingClassifier":
        for model in self.models:  # fit the ensemble models
            model.fit(dataset)

        predictions = np.array([model.predict(dataset) for model in self.models])
        self.final_model.fit(Dataset(np.transpose(predictions),dataset.y))
        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        predictions = np.array(
            [model.predict(dataset) for model in self.models])
        final_predict = self.final_model.predict(Dataset(np.transpose(predictions), dataset.y))
        return final_predict

    def score(self, dataset: Dataset) -> float:
        return round(accuracy(dataset.y, self.predict(dataset)), 4)