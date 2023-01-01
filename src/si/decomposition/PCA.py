import numpy as np
from si.src.si.data.dataset import Dataset
import pandas as pd

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset: Dataset) -> "PCA":
        self.mean = np.mean(dataset.X, axis=0) # calculo da média
        self.centered_data = dataset.X - dataset.X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(self.centered_data, full_matrices=False) # calculo do u , s e vt
        self.components = Vt[:self.n_components]
        EV = (S ** 2) / (len(dataset.X) - 1) # calculo da variancia explicada
        self.explained_variance = EV[:self.n_components] # a variancia explicada corresponde aos primeiros n_components
        return self

    def transform(self) -> np.ndarray:
        V = self.components.T
        X_reduced = np.dot(self.centered_data, V)
        return X_reduced

    def fit_transform(self, dataset: Dataset):
        self.fit(dataset)
        return self.transform()

if __name__ == "__main__":
    import pandas as pd
    dataset= pd.read_csv("C:\\Users\\ruben\\PycharmProjects\\SI\\si\\datasets\\iris.csv")
    pca = PCA(n_components=2)
    print(pca.fit_transform(dataset=dataset))

