#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('BigMart Sales Data.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


categorical_data = df.select_dtypes(include=[object])
print("count of categiorical features in the dataset: ",categorical_data.shape[1])
numerical_data = df.select_dtypes(include=[np.float64,np.int64])
print("count of numerical features in the dataset: ",numerical_data.shape[1])


# In[9]:


categorical_data.head()


# In[10]:


sns.countplot(x='Outlet_Size', data=categorical_data)


# In[11]:


categorical_data['Outlet_Size'].value_counts()


# In[12]:


categorical_data['Outlet_Size'] = categorical_data['Outlet_Size'].fillna(categorical_data['Outlet_Size'].mode()[0])


# In[13]:


categorical_data.isnull().sum()


# In[14]:


sns.countplot(x='Item_Fat_Content', data = categorical_data)


# In[15]:


categorical_data.replace({'Item_Fat_Content':{'low fat':'Low Fat', 'LF':'Low Fat','reg':'Regular'}}, inplace=True)


# In[16]:


sns.countplot(x='Item_Fat_Content', data = categorical_data)


# In[18]:


sns.countplot(y='Outlet_Identifier', data = categorical_data)


# In[19]:


categorical_data['Outlet_Identifier'].value_counts()


# In[20]:


plt.figure(figsize=(12,8))
sns.countplot(y='Item_Type', data= categorical_data)


# In[22]:


fig, axes = plt.subplots(1,2, figsize=(12,5))
sns.countplot(y='Outlet_Type', data = categorical_data, ax= axes[0])
sns.countplot(y='Outlet_Location_Type', data = categorical_data, ax= axes[1])


# In[23]:


numerical_data.describe()


# In[24]:


numerical_data['Item_Weight'].hist(bins=100)


# In[25]:


numerical_data['Item_Visibility'].hist(bins=100)


# In[26]:


numerical_data['Item_MRP'].hist(bins=100)


# In[27]:


sns.countplot(x='Outlet_Establishment_Year', data=numerical_data)


# In[29]:


plt.figure(figsize=(12,8))
sns.barplot(y='Item_Type', x='Item_Outlet_Sales', data=df)


# In[30]:


plt.figure(figsize=(7,7))
sns.barplot(x='Outlet_Size', y='Item_Outlet_Sales', data= df)


# In[31]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Location_Type','Item_Outlet_Sales', hue='Outlet_Type', data=df)
plt.legend()


# In[32]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[ ]:




