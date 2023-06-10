import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

import yfinance as yf

yf.pdr_override() # <== that's all it takes :-)

from pandas_datareader import data as pdr



def stock_prediction_procedure(symbol: str):
  # Get the stock quote
  df = pdr.get_data_yahoo(symbol, start = '2012-01-01', end='2023-12-17')

  # Create a new DF with Close Column
  data = df.filter(['Close'])

  #convert dataframe to numpy array
  np_dataset = data.values

  #train the model on specific number of rows
  training_data_len = math.ceil(len(np_dataset) * .8)

  #Scale the data
  scaler = MinMaxScaler(feature_range = (0, 1))
  scaled_data = scaler.fit_transform(np_dataset)

  # Create the training dataset

  # Create the scaled training data set
  train_data = scaled_data[0:training_data_len, :]

  #Split the data into x_train and y_train data sets
  x_train = []
  y_train = []

  for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

  #Convert the x_train and y_train to numpy arrays
  x_train, y_train = np.array(x_train), np.array(y_train)

  #Reshape the data
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
  #x_train.shape

  #Build the LSTM model
  model = Sequential()
  model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
  model.add(LSTM(50, return_sequences=False))
  model.add(Dense(25))
  model.add(Dense(1))

  #Compile the model
  model.compile(optimizer='adam', loss='mean_squared_error')

  #Train the model
  model.fit(x_train, y_train, batch_size=1, epochs=1)

  #Create the testing data set
  #Create a new array containing scaled values from index 1543 to 2003
  test_data = scaled_data[training_data_len - 60:, :]

  #Create the data sets x_test and y_test
  x_test = []
  y_test = np_dataset[training_data_len:, :]

  for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

  #Convert the data to a numpy array
  x_test = np.array(x_test)

  #Reshape the data
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

  #Get the models predicted price values
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)

  #Get the root mean squared  error (RMSE)
  rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

  #Plot the data
  train = data[:training_data_len]
  valid = data[training_data_len:]

  valid['Predictions'] = predictions
  return valid


'''
  #Get the quote
  apple_quote = pdr.get_data_yahoo('AAPL', start = '2012-01-01', end='2022-12-17')

  #Create a new dataframe
  new_dff = apple_quote.filter(['Close'])

  #Get the last 60 day closing price values and convert the dataframe to an array
  last_60_days = new_dff[-60:].values

  #Scale the data to be values between 0 and 1
  last_60_days_scaled = scaler.transform(last_60_days)

  #Create an empty list
  X_test = []

  #Append the past 60 days
  X_test.append(last_60_days_scaled)

  #Convert the X_test data set to a numpy array
  X_test = np.array(X_test)

  #Reshape the data
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

  #Get the predicted scaled price
  pred_price = model.predict(X_test)

  #undo the scaling
  pred_price = scaler.inverse_transform(pred_price)
'''

