# Importing libraries
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression


# Logistic Regression
class CustomLogisticRegression:
    def __init__(self, learning_rate: float, iterations: int):
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.num_of_training_examples = None
        self.num_of_features = None
        self.W, self.b, self.X, self.Y = None, None, None, None

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.num_of_training_examples, self.num_of_features = X.shape

        # weight initialization
        self.W = np.zeros(self.num_of_features)
        self.b = 0
        self.X = X
        self.Y = Y

        cost_list = np.zeros(self.iterations) # to keep track of costs
        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
            # cost = self.cost_f(Y,self.num_of_training_examples)
            # cost_list[i] = self.cost_f()  # could also return this list

        return self

    @staticmethod
    def sigmoid(X, W, b):
        x= (-1)*(np.matmul(X,W)+b)
        output = 1/(1+np.exp(x))
        return output

    @staticmethod
    def cost_f(self):
        H = self.sigmoid(self.X,self.W,self.b)
        
        cost = -np.sum(self.Y*np.log(H)+ (1-self.Y)*np.log(1-H))/len(self.Y)
        cost = np.squeeze(cost)   
        return cost

    # Helper function to update weights in gradient descent
    def update_weights(self):
        A = self.sigmoid(self.X,self.W,self.b) # A is our h(x)
        # calculate gradients
        difference = (A - self.Y.Outcome)
       
        difference = np.reshape(difference, self.num_of_training_examples)
        dW = np.dot(self.X.T, difference) / self.num_of_training_examples
        db = np.sum(difference) / self.num_of_training_examples

        # # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function h( x )
    def predict(self, test_X, threshold: float = 0.5):
        result = []
        probs = self.sigmoid(test_X, self.W,self.b)
        for j in probs:
            if j >= threshold:
                result.append(1) # if above the boundary, class 1
            else:
                result.append(0) #otherwise, assign class 0
        return result

    @staticmethod   
    def acc_score(y,y_hat):
        tp,tn,fp,fn = 0,0,0,0
        for i in range(len(y)):
            if y.Outcome[i] == 1 and y_hat[i] == 1:
                tp += 1
            elif y.Outcome[i] == 1 and y_hat[i] == 0:
                fn += 1
            elif y.Outcome[i] == 0 and y_hat[i] == 1:
                fp += 1
            elif y.Outcome[i] == 0 and y_hat[i] == 0:
                tn += 1
        precision = tp/(tp+fp) #division by zero when y_pred=[0,0,,,,0], idk why y_pred=o.predict() is always 0s...?
        recall = tp/(tp+fn)
        score = 2*precision*recall/(precision+recall)
        return score  #here I am calculating the well-known F1-SCORE


    @staticmethod   
    def acc_score1(y,y_hat):
        score = math.sqrt(np.sum((y.Outcome-y_hat)**2)) /len(y)
        
        return score  #here I am calculating the standard estimated error