"""
Fisher's Linear Discriminant Classification class.
Assume for now that the data is 2D, and classification for 2 classes.
"""
# TODO: Allow for arbitrary number of dimsnsions
# TODO: Return decision boundary
# TODO: Create classification feature based on decision boundary
# TODO: Write explanation of LDA in README.md

import numpy as np


class LDA:
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions

    @property
    def fishers_criterion(self, m_1, m_2, s_1, s_2):
        """Objective function to maximise."""
        return (m_1 - m_2) ** 2 / (s_1 ** 2 - s_2 ** 2)

    def compute_projected_quantity(self, vec, quantity):
        """Dot product a quantity onto a defined vector"""
        return (vec[0] * quantity[0] + vec[1] * quantity[1]) / (
            vec[0] ** 2 + vec[1] ** 2
        ) ** 0.5

    def compute_mean(self, vals):
        """Compute the mean of vals. Should return a vector array"""
        return np.mean(vals, axis=0)

    def compute_variance(self, mean, vals):
        """Compute the variance (s^2) of vals."""
        var = 0
        for val in vals:
            var = var + (val - mean) ** 2
        return var

    def compute_scatter_matrix(self, mean_vec, vals):
        """Compute the scatter matrix."""
        scatter_matrix = np.zeros((2, 2), dtype=float)
        mean_vec = np.expand_dims(mean_vec, axis=0)

        for val in vals:
            val = np.expand_dims(val, axis=0)
            scatter_matrix += np.matmul((val - mean_vec).transpose(), val - mean_vec)
        return scatter_matrix

    def compute_S_w_matrix(self, val1, val2):
        """Compute S_w matrix."""
        val1_mean = self.compute_mean(val1)
        val2_mean = self.compute_mean(val2)

        val1_scatter_matrix = self.compute_scatter_matrix(val1_mean, val1)
        val2_scatter_matrix = self.compute_scatter_matrix(val2_mean, val2)

        return val1_scatter_matrix + val2_scatter_matrix

    def fit(self, val1, val2):
        """Compute fisher's linear discriminant."""
        return np.matmul(
            np.linalg.inv(self.compute_S_w_matrix(val1, val2)),
            self.compute_mean(val1) - self.compute_mean(val2),
        )
