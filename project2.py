#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


traindata = pd.read_csv("TRAIN.csv")
traindata


# In[3]:


traindata.info()


# In[4]:


traindata.isnull().sum()


# In[5]:


traindata.describe()


# In[6]:


import plotly.express as px # to create pie charts 


# In[7]:


pie = traindata["Store_Type"].value_counts()
store = pie.index
orders = pie.values

fig = px.pie(traindata, values = orders, names = store)
fig.show()


# In[8]:


pie1 = traindata["Location_Type"].value_counts()
location = pie1.index
orders = pie1.values

fig = px.pie(traindata, values = orders, names = location)
fig.show()


# In[9]:


pie2 = traindata["Region_Code"].value_counts()
region = pie2.index
orders = pie2.values

fig = px.pie(traindata, values = orders, names = region)
fig.show()


# In[10]:


pie3 = traindata["Discount"].value_counts()
discount = pie3.index
orders = pie3.values

fig = px.pie(traindata, values = orders, names = discount)
fig.show()


# In[11]:


pie4 = traindata["Holiday"].value_counts()
holiday = pie4.index
orders = pie4.values

fig = px.pie(traindata, values = orders, names = holiday)
fig.show()


# In[12]:


traindata["Discount"] = traindata["Discount"].map({"No": 0, "Yes": 1})


# In[13]:


traindata["Store_Type"] = traindata["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})


# In[14]:


traindata["Location_Type"] = traindata["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})


# In[15]:


traindata


# In[16]:


x = (traindata[["Store_Type", "Location_Type", "Holiday", "Discount"]])


# In[17]:


y = np.array(traindata["#Order"])


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)


# In[20]:


#y = np.array(traindata["#Order"])


# In[21]:


#X_train


# In[22]:


#y


# In[23]:


#from sklearn.model_selection import train_test_split


# In[24]:


#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)


# In[25]:


import lightgbm as ltb
model = ltb.LGBMRegressor()


# In[26]:


model.fit(x_train, y_train)


# In[27]:


y_pred = model.predict(x_test)


# In[28]:


y_pred


# In[29]:


y_test


# In[30]:


import matplotlib.pyplot as plt

# Plotting the data
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')

# Adding labels and title
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.title('Actual vs. Predicted Values')
plt.legend()


# In[31]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[35]:


testdata = pd.read_csv("TEST_FINAL.csv")
testdata["Discount"] = testdata["Discount"].map({"No": 0, "Yes": 1})
testdata["Store_Type"] = testdata["Store_Type"].map({"S1": 1, "S2": 2, "S3": 3, "S4": 4})
testdata["Location_Type"] = testdata["Location_Type"].map({"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5})


# In[36]:


x_test_final = np.array(testdata[["Store_Type", "Location_Type", "Holiday", "Discount"]])


# In[37]:


y_pred_final = model.predict(x_test_final)


# In[38]:


y_pred_final


# In[39]:


data = pd.DataFrame(data = {"predicted_orders": y_pred_final.flatten()})


# In[40]:


data


# In[ ]:




