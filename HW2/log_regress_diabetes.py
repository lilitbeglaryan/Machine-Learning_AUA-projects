
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns

from logistic_regression import CustomLogisticRegression as log
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import sys
import subprocess

# implement pip as a subprocess: 
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas-profiling'])
from pandas_profiling import ProfileReport



data = 'data/diabetes.csv'
df = pd.read_csv(data)
print(df.head())

profile = ProfileReport(df, title="Profiling Report")
profile.to_file("diabetes.html")  #install html preview for viewing the .html file 
# and then type 'start diabetes.html' in the CLI

print("Shape is : "+str(df.shape)) # there are 108 observations 
# # # and 9 varaibles(features)

col_names = df.columns
print(col_names) #print the feature names

df.info() #all 108 observ-s do not have any Nan values for any of the features,
# #  all variables are numeric(int and float) , 
# # however beware! : Outcome is encoded as a numeric var but in fact is categoric

categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
# # print('The categorical variables are :', categorical)


print(df.describe()) #get some useful statistics about the variables, as outc

# # table = df.apply(isnull).apply(sum)
print(df.isnull().sum()) #0 nulls in the whole dataframe

print("\n")

label = df.iloc[:,-1] #seperate the target variable, the Outcome
print(label.head())

    
print(label.value_counts()) # occurences of labels "0" are 70 and of "1"s are 38
print("\n")
print(label.value_counts()/float(len(label))) #get the relative frequencies of this categor. var

# #explore 
print(df["Glucose"].mode()[0]) #the most frequently encountered glucose level among observants was 100

sns.pairplot(df, hue="Outcome",diag_kind="auto")
plt.show()


#logistic regression part

# x_train, x_test,y_train, y_test = train_test_split(x,y,random_state=104,test_size=0.25, shuffle=True)
train=df.sample(frac=0.8,random_state=200)
test=df.drop(train.index)
x_train = train.loc[:, df.columns != "Outcome"]
x_test = test.loc[:, df.columns != "Outcome"]
y_train = train.loc[:, ['Outcome']]
y_test = test.loc[:, ['Outcome']]

x_train.reset_index(inplace = True,drop = True)
x_test.reset_index(inplace = True,drop=True)
y_train.reset_index(inplace = True,drop=True)
y_test.reset_index(inplace = True,drop=True)
# my implementation
epoch = 10
log_reg = log(0.1, epoch)

new_obj = log_reg.fit(x_train, y_train)
# # print(new_obj.X) # the X which does not change obviosuly
print(new_obj.W) #the updated and optimized weights of the corresponding eight variables
print(new_obj.b) #the estimated bias 

# #calculate the accuracy

y_pred = new_obj.predict(x_test) #take by default 0.5 as the boundary

score = new_obj.acc_score1(y_test,y_pred)
print(score)  #  standard estimated error is 0.12026142323020866 , my accuracy is way worse than the python's



#now with the python's Logregress
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train,epochs= epoch)
pred = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)  #if predicts on the trained data, accuracy is 1.0=100%, 
# # otherwise, in this case it was 0.5925925925925926  , by default uses the mean accuracy
print(score)   








