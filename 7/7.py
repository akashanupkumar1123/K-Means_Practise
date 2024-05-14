#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('College_Data',index_col=0)


# In[3]:


df.head()


# In[ ]:





# In[4]:


df.info()


# In[ ]:





# In[5]:


df.describe()


# In[ ]:





# In[11]:


sns.set_style('whitegrid')
sns.lmplot('Outstate', 'F.Undergrad', data=df, hue='Private',
           palette='coolwarm', size=6, aspect=1, fit_reg=False)


# In[ ]:





# In[12]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# In[ ]:





# In[13]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[ ]:





# In[14]:


df[df['Grad.Rate'] > 100]


# In[ ]:





# In[15]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[ ]:





# In[16]:


df[df['Grad.Rate'] > 100]


# In[ ]:





# In[17]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[20]:


from sklearn.cluster import KMeans


# In[21]:


kmeans = KMeans(n_clusters=2)


# In[22]:


kmeans.fit(df.drop('Private',axis=1))


# In[23]:


kmeans.cluster_centers_


# In[24]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[25]:


df['Cluster'] = df['Private'].apply(converter)


# In[26]:


df.head()


# In[27]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# In[ ]:




