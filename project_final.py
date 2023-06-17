#!/usr/bin/env python
# coding: utf-8

# In[23]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import sklearn
from sklearn.decomposition import TruncatedSVD


# In[24]:


# Loading the dataset
amazon_ratings = pd.read_csv(r"C:\Users\Ozapa\AppData\Local\Temp\Temp1_archive (1).zip\ratings_Beauty.csv")
amazon_ratings = amazon_ratings.dropna()
amazon_ratings.head()


# In[25]:


amazon_ratings.shape


# In[26]:


popular_products = pd.DataFrame(amazon_ratings.groupby('ProductId')['Rating'].count())
most_popular = popular_products.sort_values('Rating', ascending=False)
most_popular.head(10)


# In[27]:


most_popular.head(30).plot(kind = "bar")


# In[28]:


# Analysis:

# The above graph gives us the most popular products (arranged in descending order) sold by the business.

# For eaxmple, product, ID # B001MA0QY2 has sales of over 7000, the next most popular product, ID # B0009V1YR8 has sales of 3000, etc.


# In[29]:


amazon_ratings1 = amazon_ratings.head(10000)


# In[30]:


# Model-based collaborative filtering system
# Recommend items to users based on purchase history and similarity of ratings provided by other users who bought items to that
# of a particular customer.
# A model based collaborative filtering technique is closen here as it helps in making predictinfg products for a particular 
# user by identifying patterns based on preferences from multiple user data.


# In[31]:


ratings_utility_matrix = amazon_ratings1.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
ratings_utility_matrix.head()


# In[32]:


# Utility Matrix based on products sold and user reviews
# Utility Matrix : An utlity matrix is consists of all possible user-item preferences (ratings) details represented as a matrix.
# The utility matrix is sparce as none of the users would buy all teh items in the list, hence, most of the values are unknown.


# In[33]:


ratings_utility_matrix.shape


# In[34]:


# Transposing the matrix


# In[35]:


X = ratings_utility_matrix.T
X.head()


# In[36]:


X.shape


# In[37]:


# Unique products in subset of data


# In[38]:


X1 = X


# In[39]:


# Decomposing the Matrix


# In[40]:


SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# In[41]:


# Correlation Matrix


# In[42]:


correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# In[43]:


# Isolating Product ID # 6117036094 from the Correlation Matrix
# Assuming the customer buys Product ID # 6117036094 (randomly chosen)


# In[44]:


X.index[99]


# In[45]:


# Index # of product ID purchased by customer


# In[46]:


i = "6117036094"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# In[47]:


# Correlation for all items with the item purchased by this customer based on items rated by other customers people who 
# bought the same product
# used to find relationship between elements


# In[48]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape


# In[49]:


# Recommending top 10 highly correlated products in sequence


# In[50]:


Recommend = list(X.index[correlation_product_ID > 0.90])

# Removes the item already bought by the customer
Recommend.remove(i) 

Recommend[0:9]


# In[51]:


# Product Id # Here are the top 10 products to be displayed by the recommendation system to the above customer based on the purchase history of other customers in the website


# In[ ]:




