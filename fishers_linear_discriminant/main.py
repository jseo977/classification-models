"""Run LDA."""

from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

from LDA import LDA

if __name__ == "__main__":
    data = load_iris()

    # Get Raw Data
    quantitative_data = data["data"]
    setosa_data = quantitative_data[data["target"] == 0]
    versicolor_data = quantitative_data[data["target"] == 1]

    # Get only sepal length and petal length (arbitrary choice).
    setosa_lengths = np.array([[vals[0], vals[2]] for vals in setosa_data])
    versicolor_lengths = np.array([[vals[0], vals[2]] for vals in versicolor_data])

    fig, ax = plt.subplots()
    ax.scatter(setosa_lengths[:, 0], setosa_lengths[:, 1])
    ax.scatter(versicolor_lengths[:, 0], versicolor_lengths[:, 1])
    plt.xlim([0, 8])
    plt.ylim([0, 8])
    # plt.show()

    lda = LDA(dimensions=3)
    fishers = lda.fit(setosa_lengths, versicolor_lengths)
    print(fishers)
    setosa_mean = lda.compute_mean(setosa_lengths)
    versicolor_mean = lda.compute_mean(versicolor_lengths)

    ax.scatter(
        [setosa_mean[0], versicolor_mean[0]],
        [setosa_mean[1], versicolor_mean[1]],
        marker="o",
        color="r",
    )
    ax.plot(
        [10 * fishers[0] + setosa_mean[0], -10 * fishers[0] + setosa_mean[0]],
        [10 * fishers[1] + setosa_mean[1], -10 * fishers[1] + setosa_mean[1]],
        "b",
    )
    print([10 * fishers[0], -10 * fishers[0]], [10 * fishers[1], -10 * fishers[1]])
    plt.show()
