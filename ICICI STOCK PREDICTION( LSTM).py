#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[8]:


from nsepy import get_history
from datetime import datetime
np.set_printoptions(suppress=True)


# In[9]:


startDate=datetime(2019, 1,1)
endDate=datetime(2020, 1, 20)

StockData=get_history(symbol='ICICIBANK', start=startDate, end=endDate)
print(StockData.shape)
StockData.head()


# In[10]:


StockData['TradeDate']=StockData.index
StockData.tail()


# In[12]:


StockData.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=40)


# In[13]:


FullData=StockData[['Close']].values
FullData[0:10]


# In[17]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc=MinMaxScaler()

DataScaler = sc.fit(FullData)
X=DataScaler.transform(FullData)

X[0:10]


# In[19]:


X_samples = list()
y_samples = list()

NumerOfRows = len(X)
TimeSteps=5


# In[20]:


for i in range(TimeSteps , NumerOfRows , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)


# In[21]:


X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
X_data.shape


# In[22]:


y_data=np.array(y_samples)
y_data=y_data.reshape(y_data.shape[0], 1)
y_data.shape


# In[23]:


TestingRecords=5

X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]


# In[24]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[25]:


for inp, out in zip(X_train[0:5], y_train[0:5]):
    print(inp,'-->', out)


# In[26]:


TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
print("Number of TimeSteps:", TimeSteps)
print("Number of Features:", TotalFeatures)


# In[27]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[28]:


regressor = Sequential()
regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures)))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[33]:


regressor.fit(X_train, y_train, batch_size = 3, epochs = 70)


# In[35]:


predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
predicted_Price


# In[36]:


orig=y_test
orig=DataScaler.inverse_transform(y_test)
orig


# In[37]:


100 - (100*(abs(orig-predicted_Price)/orig)).mean()


# In[39]:


plt.plot(predicted_Price, color = 'blue', label = 'Predicted Volume')
plt.plot(orig, color = 'blue', label = 'Original Volume')

plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(TestingRecords), StockData.tail(TestingRecords)['TradeDate'])
plt.ylabel('Stock Price')

plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()


# In[40]:


TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))

FullDataPredictions=np.append(TrainPredictions, TestPredictions)
FullDataOrig=FullData[TimeSteps:]

# plotting the full data
plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
plt.plot(FullDataOrig , color = 'blue', label = 'Original Price')


plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()


# In[ ]:




