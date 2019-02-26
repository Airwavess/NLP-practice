
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


X = np.random.rand(100, 2)


# In[3]:


X


# In[4]:


plt.scatter(X[:, 0], X[:, 1])


# In[5]:


from sklearn.cluster import KMeans


# In[6]:


clf = KMeans(n_clusters=3)


# In[7]:


clf.fit(X)


# In[8]:


clf.labels_


# In[9]:


plt.scatter(X[:, 0], X[:, 1], c=clf.labels_)

