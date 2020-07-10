#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Function to calculate INFECTED FATALITY RATE
def ifr(infected,deaths):
    ifr=(deaths/infected)*100
    print("The CMR={}%".format(ifr))


# In[3]:


ifr(136596,5732)


# In[ ]:




