#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[3]:


df = pd.read_csv("bank-full.csv")


# In[ ]:





# In[4]:


df.head()


# In[ ]:





# In[5]:


df.info()


# In[ ]:





# In[6]:


plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='age')


# In[ ]:





# In[7]:


plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='age',hue='loan')


# In[ ]:





# In[8]:


plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='pdays')


# In[ ]:





# In[9]:


plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df[df['pdays']!=999],x='pdays')


# In[ ]:





# In[10]:


plt.figure(figsize=(12,6),dpi=200)
sns.histplot(data=df,x='duration',hue='contact')
plt.xlim(0,2000)


# In[ ]:





# In[11]:


plt.figure(figsize=(12,6),dpi=200)
sns.countplot(data=df,x='previous',hue='contact')


# In[ ]:





# In[12]:


sns.countplot(data=df,x='contact')


# In[ ]:





# In[13]:


# df['previous'].value_counts()
df['previous'].value_counts().sum()-36954
# 36954 vs. 8257


# In[ ]:





# In[14]:


df.head()


# In[ ]:





# In[15]:


plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='job',order=df['job'].value_counts().index)
plt.xticks(rotation=90);


# In[ ]:





# In[16]:


plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='education',order=df['education'].value_counts().index)
plt.xticks(rotation=90);


# In[ ]:





# In[17]:


plt.figure(figsize=(12,6),dpi=200)
# https://stackoverflow.com/questions/46623583/seaborn-countplot-order-categories-by-count
sns.countplot(data=df,x='education',order=df['education'].value_counts().index,hue='default')
plt.xticks(rotation=90);


# In[ ]:





# In[18]:


sns.countplot(data=df,x='default')


# In[ ]:





# In[19]:



sns.pairplot(df)


# In[ ]:





# In[20]:


df.head()


# In[ ]:





# In[21]:


X = pd.get_dummies(df)


# In[ ]:





# In[22]:


X


# In[ ]:





# In[23]:


from sklearn.preprocessing import StandardScaler


# In[ ]:





# In[24]:


scaler = StandardScaler()


# In[ ]:





# In[25]:


scaled_X = scaler.fit_transform(X)


# In[ ]:





# In[26]:


from sklearn.cluster import KMeans


# In[ ]:





# In[27]:


model = KMeans(n_clusters=2)


# In[ ]:





# In[28]:


cluster_labels = model.fit_predict(scaled_X)


# In[ ]:





# In[29]:


# IMPORTANT NOTE: YOUR 0s and 1s may be opposite of ours,
# makes sense, the number values are not significant!
cluster_labels


# In[ ]:





# In[30]:


len(scaled_X)


# In[ ]:





# In[31]:


len(cluster_labels)


# In[ ]:





# In[32]:


X['Cluster'] = cluster_labels


# In[ ]:





# In[33]:


sns.heatmap(X.corr())


# In[ ]:





# In[34]:


X.corr()['Cluster']


# In[ ]:





# In[36]:


plt.figure(figsize=(12,6),dpi=200)
X.corr()['Cluster'].iloc[:-1].sort_values().plot(kind='bar')


# In[ ]:





# In[37]:


ssd = []

for k in range(2,10):
    
    model = KMeans(n_clusters=k)
    
    
    model.fit(scaled_X)
    
    #Sum of squared distances of samples to their closest cluster center.
    ssd.append(model.inertia_)


# In[ ]:





# In[38]:


plt.plot(range(2,10),ssd,'o--')
plt.xlabel("K Value")
plt.ylabel(" Sum of Squared Distances")


# In[ ]:





# In[39]:


ssd


# In[ ]:





# In[40]:


# Change in SSD from previous K value!
pd.Series(ssd).diff()


# In[ ]:





# In[41]:


pd.Series(ssd).diff().plot(kind='bar')


# In[ ]:




