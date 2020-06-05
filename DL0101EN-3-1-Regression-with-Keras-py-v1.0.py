#!/usr/bin/env python
# coding: utf-8

# ## Regression Model for assignment

# <a id="item31"></a>

# ## Download and Clean Dataset

# Let's start by importing the <em>pandas</em> and the Numpy libraries.

# In[1]:


import pandas as pd
import numpy as np


# We will be playing around with the same dataset that we used in the videos.
# 
# <strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:</strong>
# 
# <strong>1. Cement</strong>
# 
# <strong>2. Blast Furnace Slag</strong>
# 
# <strong>3. Fly Ash</strong>
# 
# <strong>4. Water</strong>
# 
# <strong>5. Superplasticizer</strong>
# 
# <strong>6. Coarse Aggregate</strong>
# 
# <strong>7. Fine Aggregate</strong>

# Let's download the data and read it into a <em>pandas</em> dataframe.

# In[2]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa. 

# #### Let's check how many data points we have.

# In[3]:


concrete_data.shape


# So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.

# Let's check the dataset for any missing values.

# In[4]:


concrete_data.describe()


# In[5]:


concrete_data.isnull().sum()


# The data looks very clean and is ready to be used to build our model.

# #### Split data into predictors and target

# The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.

# In[6]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# <a id="item2"></a>

# In[7]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# Let's save the number of predictors to *n_cols* since we will need this number when building our network.

# In[8]:


n_cols = predictors_norm.shape[1] # number of predictors
n_cols


# <a id="item1"></a>

# <a id='item32'></a>

# #### Let's go ahead and import the Keras library

# In[9]:


import keras


# As you can see, the TensorFlow backend was used to install the Keras library.

# Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.

# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


# <a id='item33'></a>

# In[17]:


# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# ## Build a Neural Network

# Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.

# In[12]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# The above function create a model that has two hidden layers, each of 50 hidden units.

# <a id="item4"></a>

# ## Train and Test the Network

# Let's call the function now to create our model.

# In[13]:


# build the model
model = regression_model()


# Next, we will train and test the model at the same time using the *fit* method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.

# In[15]:


# fit the model
model.fit(X_train, y_train, epochs=50,verbose=1)


# <hr>
# 
# Copyright &copy; 2019 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
