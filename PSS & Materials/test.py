import pandas as pd
# test classification dataset
from sklearn.datasets import make_classification

# evaluate adaboost algorithm for classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier

# adaboost
df = pd.read_csv("bank-full.csv", sep=';')
# print(df.columns)

from sklearn.model_selection import train_test_split

# train, test = train_test_split(df, test_size=0.2,shuffle=True,random_state=42)
# train_x,train_y = train.iloc[:, :-1], train.iloc[:, [-1]]
# test_x,test_y = test.iloc[:, :-1], test.iloc[:, [-1]]
X= df.drop("y",axis=1)
y=df["y"]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)


# print(train_x.columns)
# print(test_x.head)
model = AdaBoostClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
df.isna().sum()
df.des



