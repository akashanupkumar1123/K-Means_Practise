#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[2]:


df = pd.read_csv('cluster_mpg.csv')


# In[3]:


df = df.dropna()


# In[4]:


df.head()


# In[ ]:





# In[5]:


df.describe()


# In[ ]:





# In[6]:


df['origin'].value_counts()


# In[7]:


df_w_dummies = pd.get_dummies(df.drop('name',axis=1))


# In[8]:


df_w_dummies


# In[ ]:





# In[9]:


from sklearn.preprocessing import MinMaxScaler


# In[10]:


scaler = MinMaxScaler()


# In[11]:


scaled_data = scaler.fit_transform(df_w_dummies)


# In[12]:


scaled_data


# In[ ]:





# In[13]:


scaled_df = pd.DataFrame(scaled_data,columns=df_w_dummies.columns)


# In[14]:


plt.figure(figsize=(15,8))
sns.heatmap(scaled_df,cmap='magma');


# In[ ]:





# In[15]:


sns.clustermap(scaled_df,row_cluster=False)


# In[ ]:





# In[16]:


sns.clustermap(scaled_df,col_cluster=False)


# In[ ]:





# In[17]:


from sklearn.cluster import AgglomerativeClustering


# In[18]:


model = AgglomerativeClustering(n_clusters=4)


# In[19]:


cluster_labels = model.fit_predict(scaled_df)


# In[20]:


cluster_labels


# In[ ]:





# In[21]:


plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=df,x='mpg',y='weight',hue=cluster_labels)


# In[ ]:





# In[22]:


model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)


# In[23]:


cluster_labels = model.fit_predict(scaled_df)


# In[24]:


cluster_labels


# In[ ]:





# In[25]:


from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy


# In[ ]:





# In[26]:


linkage_matrix = hierarchy.linkage(model.children_)


# In[27]:


linkage_matrix


# In[ ]:





# In[28]:


plt.figure(figsize=(20,10))
# Warning! This plot will take awhile!!
dn = hierarchy.dendrogram(linkage_matrix)


# In[ ]:





# In[29]:


plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=48)


# In[ ]:





# In[30]:


scaled_df.describe()


# In[ ]:





# In[31]:


scaled_df['mpg'].idxmax()


# In[32]:


scaled_df['mpg'].idxmin()


# In[ ]:





# In[33]:


a = scaled_df.iloc[320]
b = scaled_df.iloc[28]
dist = np.linalg.norm(a-b)


# In[ ]:





# In[34]:


dist


# In[ ]:





# In[35]:


np.sqrt(len(scaled_df.columns))


# In[ ]:





# In[36]:


model = AgglomerativeClustering(n_clusters=None,distance_threshold=2)


# In[ ]:





# In[37]:


cluster_labels = model.fit_predict(scaled_data)


# In[ ]:





# In[38]:


cluster_labels


# In[ ]:





# In[39]:


np.unique(cluster_labels)


# In[ ]:





# In[40]:


linkage_matrix = hierarchy.linkage(model.children_)


# In[ ]:





# In[41]:


linkage_matrix


# In[ ]:





# In[42]:


plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=11)


# In[ ]:




