"""
Principal Component Analysis Class
"""

import numpy as np
from matplotlib import pyplot as plt

class PCA():
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions

    def standardise_data(self, x):
        return(x - np.mean(x))/np.std(x)

    def _standardise_data(self, x):
        return (sum([(y - sum(x)/len(x)) ** 2 for y in x])/(len(x) - 1)) ** 0.5

    def covariance(self, x, y):
        return sum([(x_i - sum(x)/len(x))*(y_i - sum(y)/len(y)) for x_i, y_i in zip(x, y)])/(len(x)-1)

    def covariance_matrix(self, data):
        """
        Assume for now that the data is in the shame (m, n), where m (rows) is the number of dimensions,
        and n (columms) is the number of observations in the data.
        """
        cov_matrix = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range (len(data)):
                cov_matrix[i, j] = self.covariance(data[i], data[j])
        return cov_matrix

    def _eigen(self, cov):
        """ Find eigenvalues and eigenvectors using default packages """
        # TODO: Actually implement this
        pass

    def _determinant(self, cov):
        """ Compute the determinant of a matrix using the cofactor method """
        # TODO: Actually implement this
        def det_recursive(mat):
            pass
    
    def eigen(self, cov):
        """ Find the eigenvalue and eigenvectors of matrix cov """
        eigenvalues, feature_vec =  np.linalg.eig(cov)
        return eigenvalues, feature_vec

    def determinant(self, cov):
        """ Find determinant of matrix cov """
        return np.linalg.det(cov)
    
    def normalise(self, data):
        return data/np.sum(data)

    def plot_scree(self, data):
        plt.bar(1 + np.array(range(len(data))), self.normalise(data))
        plt.ylabel('Percentage of variation explained')
        plt.xlabel('Eigenvalue')


    def fit(self, data, scree = False):
        standardised_data = []
        for dim in range(self.dimensions):
            standardised_data.append(self.standardise_data(data[dim]))
        standardised_data = np.array(standardised_data)
        cov_mat = self.covariance_matrix(standardised_data)

        # Sort
        eigenvalues, feature_matrix = self.eigen(cov_mat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        feature_matrix = feature_matrix[:, idx]
        if scree:
            self.plot_scree(eigenvalues)
        return np.matmul(feature_matrix, data[:self.dimensions])
