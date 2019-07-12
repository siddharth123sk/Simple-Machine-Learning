#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


import numpy as np


# In[3]:


from sklearn import datasets, linear_model


# In[4]:


from sklearn.metrics import mean_squared_error, r2_score


# In[7]:


diabetes = datasets.load_diabetes()


# In[8]:


diabetes_X = diabetes.data[:, np.newaxis, 2]


# In[9]:


diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]


# In[10]:


diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]


# In[11]:


regr = linear_model.LinearRegression()


# In[12]:


regr.fit(diabetes_X_train, diabetes_y_train)


# In[13]:


diabetes_y_pred = regr.predict(diabetes_X_test)


# In[14]:


print('Coefficients: \n', regr.coef_)


# In[17]:


print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))


# In[18]:


print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))


# In[19]:


plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:




