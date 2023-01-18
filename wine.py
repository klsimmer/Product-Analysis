#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[11]:


from matplotlib import rcParams
rcParams['figure.figsize'] = 15, 5
sns.set_style('darkgrid')


# In[12]:


wine_df = pd.read_csv('wine.csv')


# In[46]:


wine_df.head()


# In[14]:


wine_df.drop_duplicates(inplace=True)


# In[15]:


#To give equal importance to all features, we need to scale the continuous features. We will be using scikit-learn’s StandardScaler as the feature matrix is a mix of binary and continuous features 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(wine_df)

pd.DataFrame(data_scaled).describe()


# In[31]:



kmeans.fit(data_scaled)


# In[32]:


SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[44]:


kmeans = KMeans(n_jobs = -1, n_clusters = 6, init='k-means++')
kmeans.fit(data_scaled)
y_pred = kmeans.predict(data_scaled)


# In[45]:


# this feature will show the value count of points in each of the above-formed clusters
frame = pd.DataFrame(data_scaled)
frame['cluster'] = y_pred
frame['cluster'].value_counts()


# In[37]:


plt.figure(figsize = (20,10))
plt.scatter(data_scaled[y_pred == 0,0],data_scaled[y_pred == 0,1],s = 50, c = 'green', label = "High income - Less effect on quality")
plt.scatter(data_scaled[y_pred == 1,0],data_scaled[y_pred == 1,1],s = 50, c = 'blue', label = "medium income - medium effect on quality")
plt.scatter(data_scaled[y_pred == 2,0],data_scaled[y_pred == 2,1],s = 50, c = 'black', label = "Hign income - hdiumigh effect on quality")
plt.scatter(data_scaled[y_pred == 3,0],data_scaled[y_pred == 3,1],s = 50, c = 'red', label = "Less income - high effect on quality ")
plt.scatter(data_scaled[y_pred == 4,0],data_scaled[y_pred == 4,1],s = 50, c = 'pink', label = "Less income and less effect on quality")
plt.scatter(data_scaled[y_pred == 5,0],data_scaled[y_pred == 5,1],s = 50, c = 'purple', label = "Less income and medium effect on quality")
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1], s = 100, c = "yellow", label = "centroids")
plt.xlabel("Anual income(k$) -- >")
plt.ylabel("spending score out of 100 -- >")
plt.legend()
plt.show()


# In[47]:


# countplot to check the number of clusters and number of customers in each cluster
sns.countplot(y_pred)


# In[39]:


#Now I'm going to include 3 independant variables
x = wine_df[['fixed acidity','volatile acidity','citric acid']].values


# In[40]:


# find the optimal number of clusters using elbow method  -- >This is for 3 features = ['fixed acidity','volatile acidity','citric acid']
WCSS = []
for i in range(1,11):
    model = KMeans(n_clusters = i,init = 'k-means++')
    model.fit(x)
    WCSS.append(model.inertia_)
fig = plt.figure(figsize = (7,7))
plt.plot(range(1,11),WCSS, linewidth=4, markersize=12,marker='o',color = 'red')
plt.xticks(np.arange(11))
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# In[41]:


#From the above elbow method, we can see that the optimal number of clusters == 4
# finding the clusters based on input matrix "x"
model = KMeans(n_clusters = 4, init = "k-means++", max_iter = 300, n_init = 10, random_state = 0)
y_clusters = model.fit_predict(x)


# In[43]:


## 3d scatterplot to present of 3 variables
fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[y_clusters == 0,0],x[y_clusters == 0,1],x[y_clusters == 0,2], s = 40 , color = 'blue', label = "cluster 0")
ax.scatter(x[y_clusters == 1,0],x[y_clusters == 1,1],x[y_clusters == 1,2], s = 40 , color = 'orange', label = "cluster 1")
ax.scatter(x[y_clusters == 2,0],x[y_clusters == 2,1],x[y_clusters == 2,2], s = 40 , color = 'green', label = "cluster 2")
ax.scatter(x[y_clusters == 3,0],x[y_clusters == 3,1],x[y_clusters == 3,2], s = 40 , color = '#D12B60', label = "cluster 3")
ax.set_xlabel('fixed acidity -->')
ax.set_ylabel('volatile acidity-->')
ax.set_zlabel('citric acid-->')
ax.legend()
plt.show()


# In[ ]:





# In[ ]:




