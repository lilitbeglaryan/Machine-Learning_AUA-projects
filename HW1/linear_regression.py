"""
Write full code for CustomLinearRegression Class
"""
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as lg_sklearn
from sklearn.linear_model import Ridge

from typing import Union


import warnings

#suppress warnings
warnings.filterwarnings('ignore')


class CustomLinearRegression:
    def __init__(self, C: Union[float, int] = 0, random_state: int = 42):
        self.random_state = random_state
        self.C = C  # L2 regularization coefficient, you will need this in gradient computation
        self.W = None
        self.b = None

    def init_weights(self, input_size: int, output_size: int):
        """
        Initialize weights and biases

        `W` -  matrix with shapes (input_size, output_size)
        Initialize with random numbers from Normal distribution with mean 0 and std 0.01
        `b` - vector with shape (1, output_size), initialize with 0s
        """
        np.random.seed(self.random_state)
        # print(input_size)
        # print(output_size)
        self.W = (np.random.normal(0,0.01,input_size*output_size)).reshape((input_size,output_size))
        self.b = np.zeros((input_size,output_size))
        # print("w is \n ")
        # print(self.W.shape)
        # print("b is \n ")
        # print(self.b)

    def cost_f(self,error):  #not used
        ridge_reg_term = 1/(2) * (self.C * np.sum(np.square(self.W)))
 
        cost = (1/(2) * np.sum(error ** 2)) + ridge_reg_term # 
        return cost
    
    def cost_der(self,error,X): #derivative of loss function wr.to weights
        w_ = np.sum(np.dot((-1)*X.T,error)) + self.C * np.sum(self.W) #w.r.to weigths
        b_ = (-1)*np.sum(error) #gradient w.r.to b
        result = (w_,b_)
        return result
    
    def fit(self, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000, lr: float = 0.001):
        """Train model linear regression with gradient descent

        Parameters
        ----------
        X: with shape (num_samples, input_shape)
        y: with shape (num_samples, output_shape)
        num_epochs: number of interactions of gradient descent
        lr: step of linear regression

        Returns
        -------
        None
        """
        m, n = X.shape
        self.init_weights(m,y.shape[1])  # originally was (X.shape[1], y.shape[1]) , but gave index out of bound error, y.shape=(n,)
        costs = []
        for _ in range(num_epochs):
            preds = self.predict(X)
            # compute gradients without loops, only use numpy.
            # IMPORTANT don't forget to compute gradients for L2 regularization

            error = preds - y  # TODO  #the errir term
            # print(y)
            # print(preds.shape)
            deltas = self.cost_der(error,X)
            b_grad = deltas[1]
            # W_grad = lr * (X.T.dot(error))
            W_grad  = deltas[0]
            self.W = self.W - lr* W_grad
            self.b =  self.b  - lr * b_grad # TODO
            


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Do your predictions here :)
        """
        result = (np.dot(X.T,self.W) + self.b)
        # print(result)
        return  result  # TODO


if __name__ == "__main__":

    # print(10* np.sum(np.array([1,2])))
    # TODO run this part of the code to generate plot, save plot and upload to github
    #  you are free to play with parameters
    custom_l2 = CustomLinearRegression(C=10, random_state=42)
    # custom_l2.fit(np.array([[1, 1], [1, 6]]), np.array([1, 2]).reshape((2,1)),num_epochs= 2)


    # TODO also use linear regression without L2, implemented in CustomLinearRegression.
    custom_lin_reg = CustomLinearRegression() #by default when C=0, OLS


    lg_sk = lg_sklearn()
    ridge = Ridge(alpha=10)

    X, y = make_regression(1000, n_features=1, n_targets=1, random_state=42, noise=0)
    y = np.expand_dims(y, 1)

    # adding anomalous datapoint to see effect of regression
    # you are free to comment this part to see effect on normal linear data
    X = np.vstack((X, np.array([X.max() + 20])))
    y = np.vstack((y, np.array([y.max() + 10])))

    print(X.shape)
    print(y.shape)

    # # fitting models
    custom_l2.fit(X, y)
    y_hat_l2 = custom_l2.predict(X)

    custom_lin_reg.fit(X, y)
    y_hat_lin = custom_lin_reg.predict(X)

    lg_sk.fit(X, y)
    y_hat_sk = lg_sk.predict(X)

    ridge.fit(X, y)
    y_hat_ridge = ridge.predict(X)

    # # plotting models
    print(y_hat_l2)  #my predictions are Nans???? maybe because of the matrix multiplications givng Overflow values

    plt.subplot(121)
    plt.scatter(X, y)

    plt.plot(X, y_hat_l2, color="red", label="Custom L2")  #custome
    plt.plot(X, y_hat_lin, color="k", label="Custom Lin reg") #custom


    # plt.lineplot

    #python's standard Ridge regress(with L2 regulariz) and standard linear regress with OLS
    # have identical plots, so they both predicted the target similarly
    plt.subplot(122)
    plt.plot(X,y)  #don't know why there are several X s taht have multiple targets mapped to them, the line is branched
    plt.scatter(X, y_hat_sk, color="green", label="Sklearn Lin reg",alpha = 0.8,edgecolors="black",s=60,marker = "*",)  
    plt.scatter(X, y_hat_ridge, color="orange", label="Ridge")
    plt.legend()
    plt.show()
    plt.savefig("regressions.png")
