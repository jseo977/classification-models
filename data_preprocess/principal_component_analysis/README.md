# Principal Component Analysis

PCA is a statistical dimenaionsality reduction technique that finds the dimensions/axis of greatest variability and superimposes data into these lower dimensional axis.

# The steps in PCA
PCA can be generalised to have 5 steps.

## Step 1: Standardise the data
This is because the algorithm is sensitive to the initial inputs, and the range of these inputs. This is found by taking the difference of each data point and dividing by the standard deviation for that group of data.
$$z = \frac{x - \mu_{x}}{\sigma}$$

## Step 2: Compute the standardised covariance matrix
This is to find how well the (standardised) data correlates with each other. The covariance matrix is symmetric about the diagonal due to the commutative property of the covariance function, and the diagonals of the covariance is simply the variance. Positive entries in the covariance matrix implies positive correlation, and negative entries imply inverse correlations.

## Step 3: Compute Eigenvectors & Eigenvalues of the covariance matrix
This is used to help identify the principal components (new variables). The eigenvectors provide the linear combination of the direction initial variables that provide the span of the principal components. The eigenvalues identify the degree to which the eigenvectors explain the variation in the data. The % explained by an eigenvector/principal component is given by its associated eigenvalue divided by the sum of all the eigenvalues.

## Step 4: Feature Vector
Using the % of variation explained by each of the eigenvectors, we choose which/how many principal components to take to explain the variation. We create a feature vector by concatenating the vectors (horizontally).

## Step 5: Recast data into the axis of the Principal Components
Premultiply the transpose of the standardised dataset by the transpose fo the feature vector.