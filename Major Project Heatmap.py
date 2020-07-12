#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv


# In[5]:


pwd


# In[6]:


df = pd.read_csv('C:\\Users\\thinkpad\\desktop\\case_time_series.csv')
df.head()


# In[7]:


array_2d = np.linspace(1,5,12).reshape(4,3) # create numpy 2D array
 
print(array_2d) # print numpy array


# In[8]:


sns.heatmap(array_2d) # create heatmap


# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv


# In[47]:


# Load dataset
 
Case_time_series_df = pd.read_csv("C:\\Users\\thinkpad\\desktop\\Case_time_series.csv")
Case_time_series_df.head()


# In[75]:


# Load dataset
case_time_series_df = pd.read_csv("C:\\Users\\thinkpad\\desktop\\case_time_series.csv")
case_time_series_df.head()


# In[76]:


array_2d = np.linspace(1,5,12).reshape(4,3) # create numpy 2D array
 
print(array_2d) # print numpy array


# In[77]:


sns.heatmap(array_2d) # create heatmap


# In[78]:


# set country name as index and drop Country Code, Indicator Name and Indicator Code
 
case_time_series_df  = case_time_series_df.drop(columns=['Total Confirmed', 'Total Recovered', 'Total Deceased'], axis=1).set_index('Date')
case_time_series_df


# In[79]:


# Create heatmap
 
plt.figure(figsize=(16,9))
sns.heatmap(case_time_series_df)


# In[81]:


# vmin and vmax parameter set limit of colormap
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, vmin = 0, vmax = 21)


# In[83]:


# change heatmap color using cmap
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, cmap="coolwarm")


# In[84]:


# center parameter 
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, cmap="coolwarm", center = 10.0)


# In[85]:


# robust
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, robust = True)


# In[86]:


# annot (annotate) parameter
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, annot = True)


# In[61]:


# pass 2D Numpy array to annot parameter
 
annot_arr = np.arange(1,13).reshape(4,3) # create 2D numpy array with 4 rows and 3 columns
 
sns.heatmap(array_2d, annot= annot_arr)


# In[62]:


# fmp parameter - add text on heatmap cell
 
annot_arr = np.array([['a00','a01','a02'],
                      ['a10','a11','a12'],
                      ['a20','a21','a22'],
                      ['a30','a31','a32']], dtype = str)
 
sns.heatmap(array_2d, annot = annot_arr, fmt="s") # s -string, d - decimal


# In[87]:


# annot_kws parameter
 
plt.figure(figsize=(16,9))
 
annot_kws={'fontsize':10, 
           'fontstyle':'italic',  
           'color':"k",
           'alpha':0.6, 
           'rotation':"vertical",
           'verticalalignment':'center',
           'backgroundcolor':'w'}
 
sns.heatmap(case_time_series_df, annot = True, annot_kws= annot_kws)


# In[88]:


# linewidths parameter - divede each cell of heatmap
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, linewidths=4)


# In[89]:


# linecolor parameter - change the color of heatmap line
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, linewidths=4, linecolor="k")


# In[90]:


# hide color bar with cbar parameter 
  
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df, cbar = False)


# In[91]:


# change style and format of color bar with cbar_kws parameter
   
plt.figure(figsize=(14,14))
 
cbar_kws = {"orientation":"horizontal", 
            "shrink":1,
            'extend':'min', 
            'extendfrac':0.1, 
            "ticks":np.arange(0,22), 
            "drawedges":True,
           }
 
sns.heatmap(case_time_series_df, cbar_kws=cbar_kws)


# In[92]:


# multiple heatmaps using subplots
 
plt.figure(figsize=(30,10))
 
plt.subplot(1,3,1) # first heatmap
sns.heatmap(case_time_series_df,  cbar=False, linecolor="w", linewidths=1) 
 
plt.subplot(1,3,2) # second heatmap
sns.heatmap(case_time_series_df,  cbar=False, linecolor="k", linewidths=1) 
 
plt.subplot(1,3,3) # third heatmap
sns.heatmap(case_time_series_df,  cbar=False, linecolor="y", linewidths=1) 
 
plt.show()


# In[93]:


# set seaborn heatmap title, x-axis, y-axis label and font size
 
plt.figure(figsize=(16,9))
 
ax = sns.heatmap(case_time_series_df)
 
ax.set(title="Heatmap",
      xlabel="Condition",
      ylabel="Date",)
 
sns.set(font_scale=2) # set fontsize 2


# In[94]:


# Heatmap with keyword arguments (kwargs) parameter
 
plt.figure(figsize = (16,9))
 
kwargs = {'alpha':.9,'linewidth':10, 'linestyle':'--', 'rasterized':False, 'edgecolor':'w',  "capstyle":'projecting',}
 
ax = sns.heatmap(case_time_series_df,**kwargs)


# In[95]:


# sns heatmap correlation
 
plt.figure(figsize=(16,9))
 
sns.heatmap(case_time_series_df.corr(), annot = True)


# In[96]:


# Upper triangle heatmap
 
plt.figure(figsize=(16,9))
 
corr_mx = case_time_series_df.corr() # correlation matrix
 
matrix = np.tril(corr_mx) # take lower correlation matrix
 
sns.heatmap(corr_mx, mask=matrix)


# In[97]:


# Lower triangle heatmap
 
plt.figure(figsize=(16,9))
 
corr_mx = case_time_series_df.corr() # correlation matrix
 
matrix = np.triu(corr_mx) # take upper correlation matrix
 
sns.heatmap(corr_mx, mask=matrix)


# In[ ]:




