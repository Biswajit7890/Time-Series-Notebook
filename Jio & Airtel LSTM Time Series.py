#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.max_columns',500)
pd.set_option('display.max_rows',1400000)
np.set_printoptions(suppress=True)


# In[3]:


from nsepy import get_history
from datetime import datetime

startDate=datetime(2015, 1,1)
endDate=datetime(2016, 12, 31)


# In[ ]:


## use symbol for jio 'RELIANCE'


# In[4]:


StockData=get_history(symbol='BHARTIARTL', start=startDate, end=endDate)
print(StockData.shape)
StockData.head()


# In[23]:


StockData['Tradedate']=StockData.index


# In[5]:


Stock_cols=['Close','Volume']
stock_data=pd.DataFrame(StockData, columns=Stock_cols)


# In[6]:


stock_data.isnull().sum()


# In[7]:


stock_data.shape


# In[211]:


stock_data=stock_data.drop(labels=['Volume_shock'], axis=1)


# In[227]:


stock_data


# In[204]:


vol=stock_data['Volume']


# In[205]:


def volume_shock(inpdata):
    A=[]
    for i in range(0,len(inpdata)):
        z=((inpdata[i]-inpdata[i-1])/inpdata[i-1])*100
        A.append(z)
    return(A)


# In[206]:


Data=volume_shock(vol)


# In[207]:


stock_data['Volume_shock']=Data


# In[208]:


def volume_shock1(inpdata):
    if(np.round(inpdata)>0.1):
        return(1)
    else:
        return(0)


# In[209]:


stock_data['Volume_shock1']=stock_data['Volume_shock'].apply(volume_shock1)


# In[190]:


vol1=stock_data['Close']


# In[191]:


def closing_price1(inpdata):
    B=[]
    for i in range(0,len(inpdata)):
        z=((inpdata[i]-inpdata[i-1])/inpdata[i-1])*100
        B.append(z)
    return(B)
    


# In[192]:


data1=closing_price1(vol1)


# In[193]:


stock_data['Price_shocks']=data1


# In[194]:


def closing_price2(inpdata):
    if(np.round(inpdata)>0.02):
        return(1)
    else:
        return(0)


# In[195]:


stock_data['Price_shocks1']=stock_data['Price_shocks'].apply(closing_price2)


# In[196]:


def Pricing_black_swan_price1(inpdata):
    B=[]
    for i in range(0,len(inpdata)):
        z=((inpdata[i]-inpdata[i-1])/inpdata[i-1])*100
        B.append(z)
    return(B)
    


# In[197]:


data2=Pricing_black_swan_price1(vol1)


# In[198]:


stock_data['Pricing_black_swan_price1']=data2


# In[199]:


def Pricing_black_swan_price2(inpdata):
    if(np.round(inpdata)>0.05):
        return(1)
    else:
        return(0)


# In[200]:


stock_data['Pricing_black_swan_price2']=stock_data['Pricing_black_swan_price1'].apply(closing_price2)


# In[217]:


data1=stock_data['Volume_shock1']
data2=stock_data['Price_shocks1']


# In[224]:


def Pricing_shock_without_volume_shock(X,Y):
    if(X.any()==0 and Y.any()==1):
        return(1)
    else:
        return(0)


# In[225]:


Res=Pricing_shock_without_volume_shock(data1,data2)


# In[226]:


stock_data['Pricing_shock_without_volume_shock']=Res


# In[228]:


stock_data.head()


# In[230]:


FullData=StockData[['Close']].values
FullData[0:10]


# In[232]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

MX=MinMaxScaler()
X=MX.fit_transform(FullData)


# In[237]:


TimeSteps=10 ## change the time steps to change the rolling window like 20,30,40,50


# In[253]:


def Defining_Data(Data,TimeSteps):
    X_samples = list()
    y_samples = list()
    NumerOfRows = len(Data)
    for i in range(TimeSteps , NumerOfRows , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i]
        X_samples.append(x_sample)
        y_samples.append(y_sample)
    X_data=np.array(X_samples)
    X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
    print('*'*60)
    print(X_data.shape)
    y_data=np.array(y_samples)
    y_data=y_data.reshape(y_data.shape[0], 1)
    print('*'*60)
    print(y_data.shape)
    X_train=X_data[:-TimeSteps]
    X_test=X_data[-TimeSteps:]
    y_train=y_data[:-TimeSteps]
    y_test=y_data[-TimeSteps:]
    print('*'*60)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    print('*'*60)
    print("total of time steps",TimeSteps)
    print("total of Feautes",TotalFeatures)
    return(TimeSteps,TotalFeatures,X_train,X_test,y_train,y_test)   


# In[254]:


X_train=0
X_test=0
y_train=0
y_test=0
Timerecords,Tot_feautres,X_train,X_test,y_train,y_test=Defining_Data(X,TimeSteps)


# In[255]:


def LSTM_Stock_Prediction(D1,D2):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    regressor = Sequential()
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (D1,D2)))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return(regressor)  


# In[256]:


## change the units while u  change the timesteps now it is 10


# In[257]:


model=LSTM_Stock_Prediction(Timerecords,Tot_feautres)


# In[259]:


model.fit(X_train, y_train, batch_size = 5, epochs = 100)


# In[260]:


predicted_Price = model.predict(X_test)
predicted_Price = MX.inverse_transform(predicted_Price)
predicted_Price


# In[261]:


Original=y_test
Original=MX.inverse_transform(y_test)
Original


# In[263]:


100 - (100*(abs(Original-predicted_Price)/Original)).mean()


# In[271]:


StockData['TradeDate']=StockData.index


# In[273]:



plt.plot(predicted_Price, color = 'red', label = 'Predicted Stock Price')
plt.plot(Original, color = 'Green', label = 'Original Stock Price')

plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(Timerecords), StockData.tail(Timerecords)['TradeDate'])

plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()


# In[ ]:




