import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas_datareader as data 
from keras.models import load_model
import streamlit as st 

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
y_symbols = ['AFRM']
from datetime import datetime
startdate = datetime(2010,12,1)
enddate = datetime(2022,12,15)

st.title('Stock Trend Prediction')

user_input= st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_yahoo(y_symbols, start=startdate, end=enddate)

#describing Data

st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visuvalization

st.subheader('Closing Price vs Tme Chart')
fig= plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Tme Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean() #moving average of last 30
fig= plt.figure(figsize=(12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'g')
plt.plot(df.Close)
st.pyplot(fig)



# splitting data into training and testing

data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.75)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.75):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)



#load my model
model=  load_model('keras_model.h5')

# testing Part

past_100_days=data_training.tail(100)

final_df=past_100_days.append(data_testing, ignore_index=True)

input_data=scaler.fit_transform(final_df)


x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


x_test, y_test= np.array(x_test), np.array(y_test)

y_predicted= model.predict(x_test)

scaler = scaler.scale_

scale_factor= 1/scaler[0]
y_predicted=y_predicted * scale_factor
y_test=y_test * scale_factor


st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test.reshape(-1), 'b', label= 'Original Price')
plt.plot(y_predicted.reshape(-1), 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')

plt.legend()
st.pyplot(fig2)

