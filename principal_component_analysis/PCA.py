"""
Principal Component Analysis Class
"""
# TODO: More comprehensive README.md

import numpy as np

class PCA():
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions

    def standardise_data(self, x):
        return(x - np.mean(x))/np.std(x)

    def _standardise_data(self, x):
        return (sum([(y - sum(x)/len(x)) ** 2 for y in x])/(len(x) - 1)) ** 0.5

    def covariance(self, x, y):
        return sum([(x_i - sum(x)/len(x))*(y_i - sum(y)/len(y)) for x_i, y_i in zip(x, y)])/len(x)

    def covariance_matrix(self, data):
        """
        Assume for now that the data is in the shame (m, n), where m (rows) is the number of dimensions,
        and n (columns) is the number of observations in the data.
        """
        cov_matrix = np.zeros(len(data), len(data))
        for i in range(len(data)):
            for j in range (len(data)):
                cov_matrix[i, j] = self.covariance(data[i], data[j])
        return cov_matrix

    def _eigen(self, cov):
        pass

    def _determinant(self, cov):
        pass