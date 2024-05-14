#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[2]:


df = pd.read_csv('CIA_Country_Facts.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[ ]:





# In[5]:


df.describe().transpose()


# In[ ]:





# In[6]:


sns.histplot(data=df,x='Population')


# In[ ]:





# In[7]:


sns.histplot(data=df[df['Population']<500000000],x='Population')


# In[ ]:





# In[8]:


plt.figure(figsize=(10,6),dpi=200)
sns.barplot(data=df,y='GDP ($ per capita)',x='Region',estimator=np.mean)
plt.xticks(rotation=90);


# In[ ]:





# In[9]:


plt.figure(figsize=(10,6),dpi=200)
sns.scatterplot(data=df,x='GDP ($ per capita)',y='Phones (per 1000)',hue='Region')
plt.legend(loc=(1.05,0.5))


# In[ ]:





# In[10]:


plt.figure(figsize=(10,6),dpi=200)
sns.scatterplot(data=df,x='GDP ($ per capita)',y='Literacy (%)',hue='Region')


# In[ ]:





# In[12]:


plt.figure(figsize=(10,6),dpi=200)
sns.scatterplot(data=df,x='GDP ($ per capita)',y='Literacy (%)',hue='Region')


# In[ ]:





# In[13]:


sns.heatmap(df.corr())


# In[ ]:





# In[14]:


sns.clustermap(df.corr())


# In[ ]:





# In[15]:


df.isnull().sum()


# In[ ]:





# In[16]:


df[df['Agriculture'].isnull()]['Country']


# In[ ]:





# In[17]:


# REMOVAL OF TINY ISLANDS
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)


# In[ ]:





# In[18]:


df.isnull().sum()


# In[ ]:





# In[20]:


#https://stackoverflow.com/questions/19966018/pandas-filling-missing-values-by-mean-in-each-group
df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean'))


# In[ ]:





# In[21]:


df.isnull().sum()


# In[ ]:





# In[22]:


df[df['Literacy (%)'].isnull()]


# In[ ]:





# In[23]:


df.isnull().sum()


# In[ ]:





# In[24]:


df = df.dropna()


# In[25]:


X = df.drop("Country",axis=1)


# In[26]:


X = pd.get_dummies(X)


# In[27]:


X.head()


# In[ ]:





# In[28]:


from sklearn.preprocessing import StandardScaler


# In[29]:


scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)


# In[ ]:





# In[30]:


scaled_X


# In[ ]:





# In[32]:


from sklearn.cluster import KMeans


# In[33]:


ssd = []

for k in range(2,30):
    
    model = KMeans(n_clusters=k)
    
    
    model.fit(scaled_X)
    
    #Sum of squared distances of samples to their closest cluster center.
    ssd.append(model.inertia_)


# In[ ]:





# In[34]:


plt.plot(range(2,30),ssd,'o--')
plt.xlabel("K Value")
plt.ylabel(" Sum of Squared Distances")


# In[ ]:





# In[35]:


pd.Series(ssd).diff().plot(kind='bar')


# In[ ]:





# In[36]:


model = KMeans(n_clusters=3)
model.fit(scaled_X)


# In[37]:


model.labels_


# In[ ]:





# In[38]:


X['K=3 Clusters'] = model.labels_


# In[39]:


X.corr()['K=3 Clusters'].sort_values()


# In[ ]:


#----------------------------------------------------------------


# In[ ]:





# In[40]:


model = KMeans(n_clusters=15)
    
model.fit(scaled_X)
    


# In[41]:


model = KMeans(n_clusters=3)
    
model.fit(scaled_X)


# In[ ]:





# In[42]:


iso_codes = pd.read_csv("country_iso_codes.csv")


# In[43]:


iso_codes


# In[ ]:





# In[44]:


iso_mapping = iso_codes.set_index('Country')['ISO Code'].to_dict()


# In[45]:


iso_mapping


# In[ ]:





# In[46]:


df['ISO Code'] = df['Country'].map(iso_mapping)


# In[ ]:





# In[47]:


df['Cluster'] = model.labels_


# In[ ]:





# In[48]:


import plotly.express as px

fig = px.choropleth(df, locations="ISO Code",
                    color="Cluster", # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale='Turbo'
                    )
fig.show()


# In[ ]:




