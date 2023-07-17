from features_engineering import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('data/non_linear.txt', sep=',', header=None)
data.columns = ['test_1', 'test_2', 'label']



# plot_training_data()  # plot data to see initial points, they are not linearly separable in 2-D

    # # creating new features and training logistic regression

# X = map_feature(test_data.test_1.values, test_data.test_2.values, DEGREE)


    # plot_training_data()  # plot data to see initial points, they are not linearly separable in 2-D

    # # creating new features and training logistic regression
X = map_feature(data.test_1.values, data.test_2.values, DEGREE)

    # n = len(data.test_1)
    # print(np.reshape(X,(((n+ 1) * (n + 2)) / 2),))

log_reg = LogisticRegression(max_iter=400)
log_reg.fit(X[:, 1:], data.label.values)

    # # concatenating coefficient of linear regression for plotting decision boundary
log_reg.intercept_ = np.expand_dims(log_reg.intercept_, axis=1)
coefficients = np.squeeze(np.concatenate([log_reg.intercept_, log_reg.coef_], axis=1))



# log_reg.score()
    
test_data = data.sample(80)
y_pred = log_reg.predict(test_data)
plot_decision_boundary(coefficients=coefficients)
print(type(test_data.label))
score = sklearn.metrics.accuracy_score(test_data.label[:,0], y_pred)
print(score)
# # concatenating coefficient of linear regression for plotting decision boundary

plot_decision_boundary(coefficients=coefficients)