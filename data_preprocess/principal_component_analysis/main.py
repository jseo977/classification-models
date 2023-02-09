"""Run PCA."""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from PCA import PCA

if __name__ == "__main__":
    data = load_iris()

    # Get Raw Data
    quantitative_data = data["data"]
    setosa_data = quantitative_data[data["target"] == 0].T
    plt.scatter(setosa_data[0], setosa_data[1])

    # Get only sepal length and petal length (arbitrary choice).
    setosa_lengths = np.array([[vals[0], vals[2]] for vals in setosa_data])
    setosa_sepal_length = setosa_lengths[0, :]
    setosa_petal_length = setosa_lengths[1, :]

    pca = PCA(dimensions=2)
    pca = pca.fit(setosa_data, scree=False)
    plt.figure()
    plt.scatter(pca[0], pca[1])
    plt.show()
    print(pca)