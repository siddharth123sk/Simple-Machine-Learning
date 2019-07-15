#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
from sklearn.svm import SVC


# In[5]:


X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])


# In[6]:


clf = SVC(gamma='auto')


# In[7]:


clf.fit(X, y)


# In[8]:


print(clf.predict([[-0.8, -1]]))


# In[ ]:




