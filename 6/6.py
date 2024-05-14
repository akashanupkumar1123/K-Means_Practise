#!/usr/bin/env python
# coding: utf-8

# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[3]:


from sklearn.datasets import make_blobs


# In[4]:



# Create Data
data = make_blobs(n_samples=200, n_features=2, 
                           centers=4, cluster_std=1.8,random_state=101)


# In[ ]:





# In[5]:


plt.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[6]:


from sklearn.cluster import KMeans


# In[7]:


kmeans = KMeans(n_clusters=4)


# In[8]:


kmeans.fit(data[0])


# In[9]:


kmeans.cluster_centers_


# In[10]:


kmeans.labels_


# In[11]:


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')


# In[ ]:




