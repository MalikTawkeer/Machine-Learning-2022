#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.cluster import KMeans


# In[4]:


import pandas as pd


# In[5]:


from sklearn.preprocessing import MinMaxScaler


# In[6]:


from matplotlib import pyplot as plt


# In[110]:


#LOADING THE DATASET
df=pd.read_csv(r"C:\Users\91705\Documents\Eastcoast_income_dataset.csv")


# In[111]:


df.head()


# In[112]:


# VISUALIZING THE DATASET
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('AGE')
plt.ylabel('INCOME')


# In[113]:


# OBJECT CREATOIN
kmeans_model = KMeans(n_clusters=3)


# In[114]:


#training the model and predict
y_predict = kmeans_model.fit_predict(df[['Age','Income($)']])


# In[115]:


# display the prediction/clusters
y_predict


# In[117]:


#make an attrubute namly cluster in a dataset and assign each row present in a dataset a cluster name
df['cluster'] = y_predict


# In[118]:


df.head()


# In[119]:


#showing the centroids of alredy calculated clusters
kmeans_model.cluster_centers_


# In[120]:


# seperating clusters from the dataset
dataset1 = df[df.cluster==0] 
dataset2 = df[df.cluster==1]
dataset3 = df[df.cluster==2]


# In[121]:


# visualization of clusters
plt.scatter(dataset1.Age,dataset1[['Income($)']])
plt.scatter(dataset2.Age,dataset2[['Income($)']])
plt.scatter(dataset3.Age,dataset3[['Income($)']])


# In[95]:


# ACCORDING TO ABOVE SHOWEN VISULAIZATION IT IS LOOKING THAT THE CLUTERS HAVE NOT BEEN MADE WITH GOOD CORRECTly SO WE NEED FURTHER COMPUTATIONS TO PREDICT WITH GOOD ACCURACY 


# In[122]:


#min max scaler used to normalize values between 0 and 1
#create scaler object
scaler = MinMaxScaler()
#NORMALIZING income column
scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])
#NORMALIZING age column
scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[157]:


df.head()


# In[124]:


#again traning the kmeans algorithm for better accuracy
kmeans = KMeans(n_clusters=3)
y_predicted = kmeans.fit_predict(df[['Age','Income($)']])


# In[125]:


y_predicted


# In[126]:


df['cluster'] = y_predicted


# In[156]:


df.head()


# In[128]:


kmeans.cluster_centers_


# In[131]:


#SEPERATING CLUSTERS
dataset1 = df[df.cluster==0]
dataset2 = df[df.cluster==1]
dataset3 = df[df.cluster==2]


# In[134]:


#visualizing clusters
plt.scatter(dataset1.Age,dataset1[['Income($)']],color='red')
plt.scatter(dataset2.Age,dataset2[['Income($)']],color='blue')
plt.scatter(dataset3.Age,dataset3[['Income($)']],color='green')
plt.xlabel('AGE')
plt.ylabel('INCOME')
plt.legend()
#displaying clusters centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[152]:


#calculating clusters or k with albow method

#within cluster sum of square (WCSS)
sse=[] # sum of square error array
krange = range(1,10)
for k in krange:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df[['Age','Income($)']])
    sse.append(kmeans.inertia_)


# In[153]:


sse


# In[155]:


#plotting WCSS to see the elbow 
plt.plot(krange,sse)
plt.xlabel('K')
plt.ylabel('SUM OF SQUARED ERROR')


# In[ ]:




