#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np  
from sklearn.cluster import KMeans  


# In[2]:


X = np.array([[5,3],  
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])


# In[4]:


plt.scatter(X[:,0],X[:,1], label='True Position')  


# In[5]:


kmeans = KMeans(n_clusters=2)  
kmeans.fit(X)


# In[6]:


print(kmeans.cluster_centers_)  


# In[7]:


print(kmeans.labels_)  


# In[9]:


plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')  


# In[10]:


plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black')  


# In[ ]:




