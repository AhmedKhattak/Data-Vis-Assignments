#!/usr/bin/env python
# coding: utf-8

# #  Data Visualization Assignment 1

# ## Objective
# 

# Perform imputation of missing data using **mean**, **median** ,**mode**, and **K-Nearest Neighbors** Algorithm with  **Euclidian (L2 norm)** and **Manhattan (L1 norm)** distance functions and visualize results

# ## Import libraries

# In[1]:


import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import seaborn as sns
import missingno as msno
from pandas.api.types import is_numeric_dtype
from pandas import Series
from pandas import DataFrame


# ## Import data sets

# In[2]:


df1 = pd.read_csv(r"Datasets/iris.csv") # iris dataset
df2 = pd.read_csv(r"Datasets/mpg.csv")  # car milage dataset
df3 = pd.read_csv(r"Datasets/diamonds.csv") # mustakbil.com data set


# ## Description of data sets

# ### IRIS

# The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

# #### First five samples

# In[6]:


df1.head()


# #### Shape of data

# In[5]:


df1.info()


# ### MPG

# This dataset contains a subset of the fuel economy data that the EPA makes available on http://fueleconomy.gov. It contains only models which had a new release every year between 1999 and 2008 - this was used as a proxy for the popularity of the car.

# #### First five samples

# In[10]:


df2.head()


# #### Shape of data

# In[9]:


df2.info()


# ### Diamonds

# A dataset containing the prices and other attributes of almost 54,000 diamonds.

# #### First five samples

# In[14]:


df3.head()


# #### Shape of data

# In[13]:


df3.info()


# ## Check missingness of Data before nullputation

# Note: If there is no missingness introduce it artificially

# 0's show no mising value

# ### IRIS

# In[43]:


df1.isnull().sum(axis = 0)


# ### MPG

# In[33]:


df2.isnull().sum(axis=0)


# ### DIAMONDS

# In[17]:


df3.isnull().sum(axis=0)


# ## Add missingness to data

# For now only add missingess to continous variables

# In[95]:


np.random.seed(69) # set seed to make code reproduceable


# Create a function to add missingness to multiple columns randomly with minimal overlap

# In[57]:


def nullpute(dataframe: DataFrame, percent_missing = 0.20, columns = None):
    # perform early exit
    if columns is None:
        print("Columns must be given !")
        return
    for col in columns:
        if(is_numeric_dtype(dataframe[col]) != True):
            print(f'At least one column "{col}" in not numeric all columns must be numeric dtype')
            return
    dataframe = dataframe.copy() # need to copy so that this function is non mutating
    # all checks have passed now perform operations
    num_rows = len(dataframe) # num of rows of dataframe
    missing_rows = int(percent_missing * num_rows) # num of rows to remove by percent
    # random_rows = np.random.randint(low = 0, 
    #                                 high = num_rows, 
    #                                 size = missing_rows)
    # Use random choice without replacement
    random_rows_per_column = [np.random.choice(num_rows,missing_rows, 
                                              replace = False) for cols in columns]
    
    
    for x,y in zip(random_rows_per_column, columns):
        dataframe.loc[x,y] = np.nan
    # dataframe.loc[random_rows,columns] = np.nan
    print(f"{missing_rows} observations have been replaced with NaN for columns {columns}")
    return dataframe
  


# #### Nullpute data 

# In[55]:


nullified_df1 = nullpute(df1, percent_missing = 0.25 , columns = ['petal-width','sepal-length'])
nullified_df2 = nullpute(df2, percent_missing = 0.25 , columns = ['hwy','cty'])
nullified_df3 = nullpute(df3, percent_missing = 0.25 , columns = ['x','z'])


# ## Check missingness of data after nullputation

# ### IRIS

# In[50]:


nullified_df1.isnull().sum(axis = 0)


# ### MPG

# In[58]:


nullified_df2.isnull().sum(axis = 0)


# ### DIAMONDS

# In[60]:


nullified_df3.isnull().sum(axis = 0)


# ## Visualize missingness

# ### IRIS

# In[88]:


msno.matrix(nullified_df1, figsize=(14,7), sparkline = False)


# ### MPG

# In[81]:


msno.matrix(nullified_df2, figsize=(14,7), sparkline = False)


# ### DIAMONDS

# In[82]:


msno.matrix(nullified_df3, figsize=(14,7), sparkline = False)


# ## Impute missing Data

# Import imputer from scikit-learn library

# In[89]:


from sklearn.impute import SimpleImputer


# Create a function to streamline imputation

# In[112]:


def impute(dataframe: DataFrame, columns = None, **kwargs):
    # perform early exit
    if  columns is None:
        print('User must provide an sklearn SimpleImputer and columns to impute on')
        return
    
    # create inline imputer
    imp = SimpleImputer(**kwargs)
    dataframe = dataframe.copy()
    # impute each colum of dataframe and then return a copy 
    for col in columns:
        numpy_arr = dataframe[col].values
        # reshape numpy array for imputer
        # will produce something like this[[1],[2],[3]] i.e an array of arrays look at np.reshape documentation
        reshaped_numpy_arr = numpy_arr.reshape(-1,1)
        imputed_res = imp.fit_transform(reshaped_numpy_arr)     
        dataframe[col]  = pd.Series(imputed_res.reshape(-1), name = col)
    return dataframe
   


# ### IRIS

# In[113]:


arguments = { 'missing_values': np.nan, 'strategy': 'mean' }  # create argument dictionary and use median strategy
df1_imputed_mean = impute(nullified_df1, columns = ['sepal-length', 'petal-width'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'median' }  # create argument dictionary and use median strategy
df1_imputed_median = impute(nullified_df1, columns = ['sepal-length', 'petal-width'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'most_frequent' }  # create argument dictionary and use median strategy
df1_imputed_mode = impute(nullified_df1, columns = ['sepal-length', 'petal-width'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'constant' }  # create argument dictionary and use median strategy
df1_imputed_constant = impute(nullified_df1, columns = ['sepal-length', 'petal-width'] , **arguments)


# ### MPG

# In[115]:


arguments = { 'missing_values': np.nan, 'strategy': 'mean' }  # create argument dictionary and use median strategy
df2_imputed_mean = impute(nullified_df2, columns = ['cty', 'hwy'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'median' }  # create argument dictionary and use median strategy
df2_imputed_median = impute(nullified_df2, columns = ['cty', 'hwy'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'most_frequent' }  # create argument dictionary and use median strategy
df2_imputed_mode = impute(nullified_df2, columns = ['cty', 'hwy'] , **arguments)

arguments = { 'missing_values': np.nan, 'strategy': 'constant' }  # create argument dictionary and use median strategy
df2_imputed_constant = impute(nullified_df2, columns = ['cty', 'hwy'] , **arguments)


# ## Visualize imputation

# Use histograms to visualize before and after imputation process
