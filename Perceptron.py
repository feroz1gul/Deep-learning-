#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas numpy matplotlib seaborn')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# In[2]:


get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip cache purge')





# In[3]:


df = pd.read_csv('placement.csv')
print(df.shape)
df.head()


# In[4]:


df.dtypes


# In[5]:


df.isnull().sum()


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


sns.scatterplot(x=df.cgpa,y=df.resume_score,hue=df.placed)


# In[8]:


x = df.iloc[:,0:2]
y = df.iloc[:,-1]


# In[9]:


x


# In[10]:


y


# In[11]:


pip install scikit-learn


# In[12]:


from sklearn.linear_model import Perceptron
p = Perceptron()


# In[13]:


p.fit(x,y)


# In[14]:


p.coef_


# In[15]:


p.intercept_


# In[16]:


pip install mlxtend


# In[17]:


from mlxtend.plotting import plot_decision_regions


# In[18]:


plot_decision_regions(x.values, y.values, clf=p, legend= 2)


# In[ ]:




