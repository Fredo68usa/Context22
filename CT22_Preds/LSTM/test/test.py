import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime

def prepare_data(data, seq_len):
    X = []
    y = []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


# Load the data
# data = pd.read_csv('data.csv',sep=" ")
# data = pd.read_csv('data.csv')
custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
# custom_date_parser = lambda x: datetime.strptime(x, "%Y-%d-%m %H:%M:%S")
data = pd.read_csv('data.csv',parse_dates=['Date'])
# data = pd.read_csv('data.csv', parse_dates=['Date'],date_parser=custom_date_parser,sep=",")

print (data.info())
print (data)

# exit(0)
dataLen = int(input("Nbr of recs to process ? "))
data = data[:dataLen]
print (len(data))
# exit(0)
# Split the data into training and test sets
split = .8
splitLen = int(len(data) * split)
print (splitLen)
# exit(0)
# train_data = data[:500]
# train_data = data.Qty[:split]
train_data = data.Qty[:splitLen]
# train_data = data[:split]
# test_data = data[500:]
test_data = data.Qty[splitLen:]
# test_data = data[split:]

# Normalize the data
mean = train_data.mean()
std = train_data.std()
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Prepare the data for LSTM
seq_len = 10

# train_X, train_y = get_train_test_sets(train_data.values, seq_len, train_frac=.9)
train_X, train_y = prepare_data(train_data.values, seq_len)
print ("train_X shape ", train_X.shape)
exit(0)
test_X, test_y = get_train_test_sets(test_data.values, seq_len ,  train_frac=.9)
# print(train_X, train_y)
print(train_X.shape)
print(train_X)
# print(test_X, test_y)
exit(0)
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(seq_len, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model
epochs = 10
batch_size = 4
verbose = 0
print (train_X.shape)
print (test_X.shape)
exit(0)
# model.fit(train_X, train_y, epochs=50, batch_size=32, verbose=2)
print (" BEFORE fit ")
model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
print (" AFTER fit ")


# Make predictions for the test set
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))
test_predictions = model.predict(test_X)

# Compute predictions for the steps beyond the test set
last_sequence = np.array(test_X[-1])
predictions = []
print("last_sequence 1 \n" ,last_sequence)
for i in range(100):
    prediction = model.predict(last_sequence.reshape(1, seq_len, 1))[0][0]
    print (" prediction " , predictions)
    predictions.append(prediction)
    last_sequence = np.concatenate((last_sequence[1:], [prediction]))
    print(" last_sequence " , i , " " , last_sequence)

# Denormalize the predictions
predictions = np.array(predictions) * std + mean

# Print the predictions
print(predictions)

