#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd


# In[9]:


#Load the dataset
Breast = pd.read_csv('/Users/teneshasmith/Downloads/Breast_Cancer.csv')


# In[10]:


Breast


# In[11]:


Breast.isna().sum()


# In[20]:


Breast['Marital Status'].value_counts()


# In[33]:


Breast['Regional Node Examined'].value_counts()


# In[34]:


Breast['Reginol Node Positive'].value_counts()


# In[35]:


Breast['Status'].value_counts()


# In[32]:


Breast['Progesterone Status'].value_counts()


# In[31]:


Breast['Estrogen Status'].value_counts()


# In[30]:


Breast['Tumor Size'].value_counts()


# In[29]:


Breast['Grade'].value_counts()


# In[28]:


Breast['differentiate'].value_counts()


# In[27]:


Breast['N Stage'].value_counts()


# In[26]:


Breast['A Stage'].value_counts()


# In[36]:


Breast['Age'].value_counts()


# In[23]:


Breast['Grade'].value_counts()


# In[24]:


Breast['Survival Months'].value_counts()

