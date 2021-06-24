#!/usr/bin/env python
# coding: utf-8

# In[99]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[114]:


#load the data
uploaded = pd.read_csv(r"C:\Users\Acer\Desktop\Stock-MArket-Forecasting-master\coin_Ethereum.csv")


# In[115]:


#store the data in the data frame
df = pd.read_csv('coin_Ethereum.csv')
df.head()


# In[116]:


#get  the no of trading days
df.shape


# In[127]:


#visualize the close price data
plt.figure(figsize=(16,8))
plt.title('Ethereum')
plt.xlabel('Days')
plt.ylabel('Close Price in USD ($')
plt.plot(df['Close'])
plt.show()


# In[118]:


#get the close price
df = df[['Close']]
df.head(4)


# In[119]:


#create a variable to predict 'x' days out into the future
future_days = 25
#create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['Close']].shift(-future_days)
df.head(4)


# In[120]:


#create the feature dataset (X) and convert it into a numpy arr and remove the last 'x' rows/days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]
print(X)


# In[121]:


#create the target data set (y) and convert it into numpy array and get the target values except the last 'x' rows
y = np.array(df['Prediction'])[:-future_days]
print(y)


# In[122]:


#split the data 75% training and 25% testing 
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)


# In[123]:


#create the models
#create the decision tree regressor model
tree = DecisionTreeRegressor().fit(x_train, y_train)
#Create the linear regression model 
lr = LinearRegression().fit(x_train, y_train)


# In[110]:


#get the last 'x' rows of the feature dataset
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[124]:


#show the model tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
#show the model linear regression prediction
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# In[129]:


#visualize the data
predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Prediction'] = predictions
plt.figure(figsize=(16,20))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price in US ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['Orig','Val', 'Pred'])
plt.show()


# In[126]:


#visualize the data
predictions = lr_prediction

valid = df[X.shape[0]:]
valid['Prediction'] = predictions
plt.figure(figsize=(16,16))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price in US ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','Prediction']])
plt.legend(['Orig','Val', 'Pred'])
plt.show()


# In[ ]:




