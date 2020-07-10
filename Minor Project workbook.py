#!/usr/bin/env python
# coding: utf-8

# In[1]:


pwd


# In[7]:


import pandas as pd

data = pd.read_csv(r'C:\\Users\\thinkpad\\desktop\\state_wise.csv') #for an earlier version of Excel use 'xls'
df = pd.DataFrame(data, columns = ['State','Confirmed','Recovered','Deaths','Active',...])

print (df)


# In[ ]:




