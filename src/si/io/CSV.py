import pandas as pd
import numpy as np

from si.src.si.data.dataset import Dataset


def read_csv(filename: str,
             sep: str = ',',
             features: bool = False,
             label: bool = False) -> Dataset:
    """
    Reads a csv file (data file) into a Dataset object
    Parameters
    ----------
    filename : str
        Path to the file
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    Returns
    -------
    Dataset
        The dataset object
    """
    data = pd.read_csv(filename, sep=sep)

    if features and label:
        features = data.columns[:-1]
        label = data.columns[-1]
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()

    elif features and not label:
        features = data.columns
        X = data.to_numpy()
        y = None

    elif not features and label:
        X = data.iloc[:, :-1].to_numpy()
        y = data.iloc[:, -1].to_numpy()
        features = None
        label = None

    else:
        X = data.to_numpy()
        y = None
        features = None
        label = None

    return Dataset(X, y, features=features, label=label)


def write_csv(filename: str,
              dataset: Dataset,
              sep: str = ',',
              features: bool = False,
              label: bool = False) -> None:
    """
    Writes a Dataset object to a csv file
    Parameters
    ----------
    filename : str
        Path to the file
    dataset : Dataset
        The dataset object
    sep : str, optional
        The separator used in the file, by default ','
    features : bool, optional
        Whether the file has a header, by default False
    label : bool, optional
        Whether the file has a label, by default False
    """
    data = pd.DataFrame(dataset.X)

    if features:
        data.columns = dataset.features

    if label:
        data[dataset.label] = dataset.y

    data.to_csv(filename, sep=sep, index=False)

#Ex1.1
ds = read_csv("C:\\Users\\ruben\\PycharmProjects\\SI\\si\\datasets\\iris.csv", ",", features = True, label = True)

#Ex1.2
primeira_variável_independente = ds.X[:, 0] # X[:, 0] seleciona tudo ate à primeira variável(0), ou seja, só seleciona a primeira
print('O valor da primeira variável independente é: ', primeira_variável_independente.shape[0])

print("----------------")

#Ex1.3
five_samples = ds.X[-5:] # seleciona os ultimos 5 (seleciona os primeiros 5 elementos a contar do fim)
mean_five_samples = np.nanmean(five_samples, axis=0)
print(mean_five_samples)

print("----------------")

#Ex1.4
amostras_maior_que_1 = np.all(ds.X > 1, axis=1)
amostras_maior_que_1 = ds.X[amostras_maior_que_1, :]
print(amostras_maior_que_1.shape)

print("------------")

#Ex1.5
iris_setosa = ds.y == "Iris-setosa" # obtém apenas os registos da iris-setosa
amostras_iris_setosa = ds.X[iris_setosa, :]
print('O número de amostras é: ', amostras_iris_setosa.shape[0])
