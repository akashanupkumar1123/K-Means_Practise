#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
clf2 = RandomForestClassifier(random_state=0, n_estimators=100)
clf3 = SVC(random_state=0, probability=True, gamma='auto')
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
X = X[:,[0, 2]]

# Plotting Decision Regions
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()


# In[ ]:





# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
sns.set()


# In[ ]:





# In[4]:


from sklearn.datasets.samples_generator import make_blobs


# In[5]:


X, y = make_blobs(n_samples=600, centers=5,
                  cluster_std=0.60, random_state=42)


# In[6]:


plt.scatter(X[:, 0], X[:, 1], s=10);


# In[ ]:





# In[7]:


from scipy.cluster.hierarchy import ward, dendrogram, linkage
np.set_printoptions(precision=4, suppress=True)


# In[8]:


distance = linkage(X, 'ward')


# In[9]:


plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance,
           leaf_rotation=90.,
           leaf_font_size=9.,);


# In[ ]:





# In[10]:


plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance, orientation="left",
           leaf_rotation=90.,
           leaf_font_size=9.,);


# In[ ]:





# In[11]:


plt.figure(figsize=(25,10))
plt.title("Hierachical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Ward's distance")
dendrogram(distance,
           leaf_rotation=90.,
           leaf_font_size=9.,);
plt.axhline(25, c='k');


# In[ ]:





# In[12]:


plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Index')
plt.ylabel("Ward's distance")
dendrogram(distance, truncate_mode='lastp',
           p=6, leaf_rotation=0., leaf_font_size=12.,
           show_contracted=True);


# In[ ]:





# In[13]:


from scipy.cluster.hierarchy import fcluster
max_d = 25
clusters = fcluster(distance, max_d, criterion='distance')
clusters


# In[ ]:





# In[14]:


plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism');


# In[ ]:





# In[15]:


k = 5
clusters = fcluster(distance, k, criterion='maxclust')


# In[ ]:





# In[16]:


plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism');


# In[ ]:





# In[17]:


from sklearn.cluster import KMeans


# In[18]:


kmeans = KMeans(n_clusters=9)
kmeans.fit(X)


# In[19]:


y_kmeans = kmeans.predict(X)


# In[20]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=10, cmap='inferno')

centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c='green', s=500, alpha=0.7);


# In[21]:


from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X, y, clf=kmeans);


# In[22]:


kmeans.inertia_


# In[23]:


sse_ = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse_.append([k, kmeans.inertia_])


# In[24]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# In[25]:


from sklearn.metrics import silhouette_score


# In[26]:


sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse_.append([k, silhouette_score(X, kmeans.labels_)])


# In[27]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# In[ ]:





# In[ ]:





# In[28]:


from sklearn.cluster import MeanShift, estimate_bandwidth


# In[29]:


from itertools import cycle


# In[30]:


bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))


# In[31]:


meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)


# In[32]:


cluster_centers = meanshift_model.cluster_centers_


# In[33]:


print('\nCenters of clusters: \n', cluster_centers)


# In[34]:


labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print('\nNumber of clusters in input data =', num_clusters)


# In[35]:


plt.figure(figsize=(10,8))
markers = '*vosx'
for i, marker in zip(range(num_clusters), markers):
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='orange')
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',  
             markerfacecolor='black', markeredgecolor='black',  
             markersize=15) 
plt.title('Clusters');

