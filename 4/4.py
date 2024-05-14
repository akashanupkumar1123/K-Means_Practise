#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


image_as_array = mpimg.imread('palm_trees.jpg')


# In[ ]:





# In[3]:


image_as_array # RGB CODES FOR EACH PIXEL


# In[ ]:





# In[4]:


plt.figure(figsize=(6,6),dpi=200)
plt.imshow(image_as_array)


# In[ ]:





# In[5]:


image_as_array.shape
# (h,w,3 color channels)


# In[ ]:





# In[ ]:


#Convert from 3d to 2d
#Kmeans is designed to train on 2D data (data rows and feature columns), 
#so we can reshape the above strip by using (h,w,c) ---> (h * w,c)


# In[6]:


(h,w,c) = image_as_array.shape


# In[7]:


image_as_array2d = image_as_array.reshape(h*w,c)


# In[ ]:





# In[8]:


from sklearn.cluster import KMeans


# In[ ]:





# In[9]:


model = KMeans(n_clusters=6)


# In[10]:


model


# In[ ]:





# In[11]:


labels = model.fit_predict(image_as_array2d)


# In[ ]:





# In[12]:


labels


# In[ ]:





# In[13]:


# THESE ARE THE 6 RGB COLOR CODES!
model.cluster_centers_


# In[ ]:





# In[14]:


rgb_codes = model.cluster_centers_.round(0).astype(int)


# In[ ]:





# In[15]:


rgb_codes


# In[ ]:





# In[16]:


quantized_image = np.reshape(rgb_codes[labels], (h, w, c))


# In[ ]:





# In[17]:


quantized_image


# In[ ]:





# In[18]:


plt.figure(figsize=(6,6),dpi=200)
plt.imshow(quantized_image)


# In[ ]:




