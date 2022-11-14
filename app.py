import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
# from keras.layers import Dense, Dropout, LSTM
# from keras.models import Sequential

start = '2010-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input,'yahoo',start,end)


#describing data
st.subheader('Data form 2010-2019')
st.write(df.describe())

#visualizations
st.subheader('Closing prise vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing prise vs Time chart WITH 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing prise vs Time chart WITH 100MA && 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)



# Splitting Data into traing and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


# #spliting data into x_train and y_train
# x_train = []
# y_train = []
# for i in range(100,data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100: i])
#     y_train.append(data_training_array[i, 0])

# x_train, y_train = np.array(x_train),np.array(y_train)


#Load my model 
model = load_model('keras_model.h5')

