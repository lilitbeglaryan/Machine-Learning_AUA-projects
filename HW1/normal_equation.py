import numpy as np


def linear_regression(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Fit linear regression to the data of form Y = w * X
    **Note** that there is not separate term for the intercept.

    Example
    >>> linear_regression(np.array([[1, 1], [1, 6]]), np.array([1, 2]))
    array([0.8, 0.2])
    """


    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T,X)) ,  X.T) , Y)
    
    ... # TODO
    return w




# if( __name__ == "__main__"):
#     X=np.array([[1, 1], [1, 6]])
    
#     Y=np.array([1, 2])
#     w= linear_regression(X,Y)
#     # print(np.matmul(X.T,X))
#     print(w)

