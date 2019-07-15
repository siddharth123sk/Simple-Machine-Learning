#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.datasets import load_iris


# In[23]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# In[24]:


iris = load_iris()


# In[25]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
tree.plot_tree(clf.fit(iris.data, iris.target))


# In[26]:


import graphviz
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data)
graph.render("iris")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




