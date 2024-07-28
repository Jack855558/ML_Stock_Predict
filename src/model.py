#Handle os Error
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


#Libraries
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
from datetime import datetime


# Get today's date
end_date = datetime.today().strftime('%Y-%m-%d')

# Fetch NVIDIA stock data up to the current date
df = yf.download('NVDA', start='2022-01-01', end=end_date)

#show dataFrame
# print(df)

#show columns and rows
# print(df.shape)

#visualize closing price history
# plt.figure(figsize=(16,8))
# plt.title('Close Price History')
# plt.plot(df['Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price USD($)', fontsize=18)
# plt.show()

#create new data frame with only close column
data = df.filter(['Close'])
#convert dataframe to numpy array
dataset = data.values 
#Get number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)
# print(training_data_len)

#Scale the data 
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
# print(scaled_data)

#Create training data with scaled dataset
train_data = scaled_data[0:training_data_len, :]
#split data into x_train (Train Model) and y_train (Test Results)
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# convert x_train and y_train to numpy arrays for training the LTSM model
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshape the data sets because LTSM expects data to be 3D but its 2D right now
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# print(x_train.shape)

#Build Long Term Short Memory / (LSTM) Model
model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create Test Data Set
#Create new array of scaled data for the test
test_data = scaled_data[training_data_len - 60: , :]
#create data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

#convert data to a numpy array
x_test = np.array(x_test)

#Reshape the data to make it 3D for the LSTM Model
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Get the models predicted price values for x_test dataset
predictions = model.predict(x_test)
#Unscale the data for the prediction
predictions = scaler.inverse_transform(predictions)

#Get the root mean square error or (RMSE) measures how accurate the model is
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD $', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
# plt.show()

#show the valid and predicted prices
print(valid)

# Fetch NVIDIA stock data up to the current date
stock_quote = yf.download('NVDA', start='2022-01-01', end=end_date)
#create new data frame
new_df = stock_quote.filter(['Close'])
#Get the last 60 close price values from the dataframe to an array
last_60_days = new_df[-60:].values
#Scale data to be values between 0-1
last_60_days_scaled = scaler.transform(last_60_days)
#Create an empty list
X_test = []
#Append past 60 days
X_test.append(last_60_days_scaled)
#Convert the x_test dataset to a numpy array
X_test = np.array(X_test)
#Reshape data to be 3d for LSTM Model
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Get Predicted scaled price
pred_price= model.predict(X_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print(f"The price of the NVIDIA stock is predicted to be {pred_price}")
