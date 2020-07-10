#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[ ]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import pandas as pd

data = pd.read_csv(r'C:\\Users\\thinkpad\\desktop\\state_wise.csv') #for an earlier version of Excel use 'xls'
df = pd.DataFrame(data, columns = ['State','Confirmed','Recovered','Deaths','Active',...])

print (df)


# In[8]:


df = pd.DataFrame(data, columns = ['State','Confirmed','Recovered','Deaths','Active',...])
df.mean()


# In[17]:


df = pd.DataFrame(data, columns = ['State','Confirmed','Recovered','Deaths','Active',...])
df.head()


# In[21]:


df.hist(column='Confirmed', bins=50)


# In[10]:


df = pd.DataFrame(data, columns = ['State','Confirmed','Recovered','Deaths','Active',...])
df.std() #standard deviation


# In[13]:


df = pd.read_csv("C:\\Users\\thinkpad\\desktop\\state_wise.csv")
df.head()


# In[22]:


df.columns


# In[24]:


X = df[['State', 'Confirmed','Recovered', 'Deaths', 'Active']] .values  #.astype(float)
X[0:5]


# In[25]:


y = df['Active'].values
y[0:5]


# In[26]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# In[ ]:




