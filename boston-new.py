#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


# In[2]:


df = pd.read_csv("boston_housing.csv")


# In[3]:


df


# In[4]:


df.head(10).T


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df[df.MEDV<750000].plot(kind="scatter",x = "LSTAT",y="PTRATIO",c="MEDV",cmap="coolwarm",s=3,figsize=(10,10))


# In[9]:


scaler = preprocessing.StandardScaler()
df[['RM','LSTAT','PTRATIO','MEDV']] = scaler.fit_transform(df[['RM','LSTAT','PTRATIO','MEDV']])


# In[10]:


df1 = df[df.MEDV<750000]
x = df1.drop(['MEDV'],axis=1)
y = df1.MEDV
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[11]:


lm = LinearRegression()
lm.fit(x_train,y_train)
y_pred = lm.predict(x_test)


# In[12]:


plt.figure(figsize=(10,10))
plt.scatter(y_test,y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red",linewidth=3)
plt.title("Real Price Vs Prediction")
plt.xlabel("Real Prices")
plt.ylabel("Predicted Prices")


# In[ ]:




