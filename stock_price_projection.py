import math
import pandas_datareader as web
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# get the stock quote
stock_ticker = input ('Enter Ticker: ')
data_from = input ('Training Data Start Date: ')
data_to = input('Training Date End Date: ')
df = web.DataReader(stock_ticker, data_source ='yahoo', start=data_from, end=data_to)
#print(df)

#the numbers of rows and columns in datasets
#print(df.shape)

# visualizing the closing price history
#plt.figure(figsize=(16,8))
#plt.title('Closing Price History')
#plt.plot(df['Close'])
#plt.xlabel('Date', fontsize=15)
#plt.ylabel('Close Price in USD($)', fontsize=15)
#plt.show()

#create a new dataframe with only close price
data = df.filter(['Close'])
dataset = data.values   #dataframe to numpy array
data_percentage = input('% of training data: ')
training_data_len = math.ceil(len(dataset) * float(data_percentage))

#scale the input data before neural network
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data =scaler.fit_transform(dataset)
#print(scaled_data)

#create training data set
train_data = scaled_data[0:training_data_len, :] #scaled training data set
#print(train_data)
#split data into x_train and y_train data set
x_train = list()  #independent training variables
y_train = list()  #dependent variables
step_size = int(input('Training data step size: '))
for i in range(step_size, len(train_data)):
    x_train.append(train_data[(i-step_size):i, 0])
    y_train.append(train_data[i,0])
#convert x_train and y_train to numpy arrays for LSTM model
x_train, y_train = np.array(x_train), np.array(y_train)
#reshape data from 2D to 3D (# of sample, # of time steps, # of features )
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
#print(x_train.shape)

#build LSTM model
model = Sequential()
model.add(LSTM(50, #neurons
               return_sequences=True, #becuase the sequential layer exist
               input_shape=(x_train.shape[0], 1))) #1st layer
model.add(LSTM(50, return_sequences=False)) #2nd layer
model.add(Dense(25)) #25 neurons in 1st dense layer
model.add(Dense(1))  #1 neuron in 2nd dense layer

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, batch_size=1, #total trainning examples/batch
                            epochs=1)#iteration when dataset pass for&back ward thru a neuron


#create testing dataset
test_data = scaled_data[training_data_len-step_size:, :]
x_test = list()
y_test = dataset[training_data_len:, :] #actual values
for i in range(step_size, len(test_data)):
    x_test.append(test_data[i-step_size:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) #reshape dataset

#get the model predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #unscaling the value
#model evaluation thru root mean squred error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#prediction
lastest_data = data[-step_size:].values
scaled_lastest_data = scaler.transform(lastest_data)
X_test = list()
X_test.append(scaled_lastest_data)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pre_price=model.predict(X_test)
pre_price=scaler.inverse_transform(pre_price)
print('Predicted Close Price is $', pre_price)

#plot
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions # add a column
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Close Price in USD($)', fontsize=15)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
plt.show()

#show valid and predicted prices
#print(valid)
