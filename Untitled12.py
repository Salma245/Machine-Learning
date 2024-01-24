#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[9]:


import pandas as pd
data=pd.read_csv('apple_quality.csv')
display(data)


# In[22]:


import numpy as np
data=data.drop('A_id',axis=1)
display(data)


# In[24]:


data=data.drop(data.index[4000])
display(data)


# In[28]:


x=data[['Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness','Acidity']]
display(x)


# In[29]:


y=data['Quality']
display(y)


# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y)


# In[31]:


display(x_train,x_test,y_train,y_test)


# In[32]:


from sklearn.tree import DecisionTreeClassifier


# In[33]:


clf=DecisionTreeClassifier()


# In[35]:


clf=clf.fit(x_train,y_train)


# In[36]:


y_pred=clf.predict(x_test)


# In[37]:


print(y_pred)


# In[38]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[39]:


from sklearn.metrics import accuracy_score


# In[40]:


print(accuracy_score(y_test,y_pred))


# In[43]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


# In[ ]:




