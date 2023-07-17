import numpy as np

from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def generate_data(num_samples, num_features):
    """
    Generates a synthetic dataset with a large number of features and a small number of samples using
    a Gaussian distribution.

    Parameters
    ----------
    num_samples : int
        The number of samples to generate.
    num_features : int
        The number of features for each sample.

    Returns
    -------
    X : ndarray of shape (num_samples, num_features)
        The input features for each sample.
    y : ndarray of shape (num_samples,)
        The output labels for each sample.
    """
    # Generate input features with a Gaussian distribution.
    X = np.random.randn(num_samples, num_features)

    # Generate random output labels.
    y = np.random.randint(0, 2, size=num_samples)

    return X, y


class KNNClassifier:
    """
    K-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, optional (default=5)
        Number of neighbors to use for classification.
    metric : str or callable, optional (default='euclidean')
        Distance metric to use for computing the distances between samples.
        Supported metrics include 'euclidean', 'manhattan', 'chebyshev', and
        any other metric supported by the `scipy.spatial.distance.cdist` function.
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        """
        Fit the KNN classifier to the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
        y : ndarray of shape (n_samples,)
            The target values.
        """

        self.X_tr = X
        self.y_tr = y
    
        # raise NotImplementedError

    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        y_pred = []
        for row in X:
            dists = [self.dist(row, x_train_row) for x_train_row in self.X_tr]
            k_neigh = np.argsort(dists)[:self.n_neighbors]
            k_classes = [self.y_tr[idx] for idx in k_neigh]
            y_pred.append(np.argmax(np.bincount(k_classes))) #return the dominating label
        return y_pred
        # raise NotImplementedError

    def euclid_dist(self, row1, row2):
        sum_squared_distance = np.sum((row1 - row2)**2)
        return np.sqrt(sum_squared_distance)
    def manhattanDistance(self, vector1, vector2):
        if len(vector1) != len(vector2):
                raise ValueError("Undefined for sequences of unequal length.")
        return np.abs(np.array(vector1) - np.array(vector2)).sum()

    def chebyshevDistance(self, vector1, vector2):
        if len(vector1) != len(vector2):
                raise ValueError("Undefined for sequences of unequal length.")
        return np.max(np.abs(np.array(vector1) - np.array(vector2)))




def evaluate_knn_performance(X, y, estimator, k_values, n_folds=5, metric='euclidean'):
    """Evaluates the performance of the KNN classifier using k-fold cross-validation.

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,)
        The target values.
    estimator :
        Your or sklearn's model
    k_values : array-like
        The values of k to use for the KNN classifier.
    n_folds : int, optional (default=5)
        The number of folds to use for cross-validation.
    metric : str, optional (default='euclidean')
        The distance metric to use for the KNN classifier.

    Returns:
    --------
    accuracies : dict
        A dictionary containing the accuracy of the KNN classifier for each value of k on validation set.
        You can interpret this as average value.

    NOTE: you can return other values if you need to.
    """
  

def plot_learning_curve(k, *args, **kwargs):
    """
    Plot the learning curve for a KNN classifier using different values of k.

    NOTE: You can modify arguments as you like, except `k`.

    Parameters:
    ----------
    k : int or array like

    Returns:
    -------
    None
    """
    raise NotImplementedError


if __name__ == "__main__":
    X, y = generate_data(100,1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

    knn = KNNClassifier()
    knn.fit(X, y)


