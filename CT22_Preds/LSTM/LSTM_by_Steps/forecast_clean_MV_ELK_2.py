##############################################
# multivariate multi-step encoder-decoder lstm
#
# 01/15/23 Version with Read off Elastic
#
# 01/27/23 Version with 
##############################################
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

import pandas as pd
from pandas import Series
import time
from datetime import datetime
import json
from elasticsearch import Elasticsearch, helpers
import sys
import numpy as np


class Forecast :

    def __init__(self, param_jason):
       print ("init")
       # self.es = Elasticsearch()
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.preds = []

       self.path = self.param_data["path"]
       self.pathlength = len(self.path)
       self.pathProcessed= self.param_data["pathProcessed"]
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       # es = Elasticsearch(['http://localhost:9200'], http_auth=('user', 'pass'))
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")

       self.fullPredsMany=[]

    # ---- getting the TS
    def getTS(self):
        print ("Getting the Time Series")
        ts_ready_tmp = p1.es.search(index="power_consumption_days", body={"query": {"match_all" : {}}, "size" : 2000})

        dataset = ts_ready_tmp['hits']['hits']
        df = pd.json_normalize(dataset)
        df = df.drop(['_index', '_type','_id','_score','_source.timestamp'], axis=1)
        df = df.rename(columns={"_source.datetime": "datetime", "_source.Global_active_power": "Global_active_power", "_source.Global_reactive_power": "Global_reactive_power","_source.Voltage":"Voltage" , "_source.Global_intensity":"Global_intensity" , "_source.Sub_metering_1":"Sub_metering_1" , "_source.Sub_metering_2":"Sub_metering_2" ,"_source.Sub_metering_3":"Sub_metering_3" ,"_source.sub_metering_4":"sub_metering_4"}, errors="raise")
        df.set_index("datetime", inplace=True)
        return(df)

    # split a univariate dataset into train/test sets
    def split_dataset(self,data):
       # split into standard weeks
       train, test = data[1:-328], data[-328:-6]
       # restructure into windows of weekly data
       train = array(split(train, len(train)/7))
       test = array(split(test, len(test)/7))
       return train, test

    # evaluate one or more weekly forecasts against expected values
    def evaluate_forecasts(self,actual, predicted):
       scores = list()
       # calculate an RMSE score for each day
       for i in range(actual.shape[1]):
            # calculate mse
            mse = mean_squared_error(actual[:, i], predicted[:, i])
            # calculate rmse
            rmse = sqrt(mse)
            # store
            scores.append(rmse)
       # calculate overall RMSE
       s = 0
       for row in range(actual.shape[0]):
         for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col])**2
       score = sqrt(s / (actual.shape[0] * actual.shape[1]))
       return score, scores

    # summarize scores
    def summarize_scores(self,name, score, scores):
       s_scores = ', '.join(['%.1f' % s for s in scores])
       print('%s: [%.3f] %s' % (name, score, s_scores))

    # convert history into inputs and outputs
    def to_supervised(self,train, n_input, n_out=7):
       # flatten data
       data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
       X, y = list(), list()
       in_start = 0
       # step over the entire history one time step at a time
       for _ in range(len(data)):
          # define the end of the input sequence
          in_end = in_start + n_input
          out_end = in_end + n_out
          # ensure we have enough data for this instance
          if out_end <= len(data):
             X.append(data[in_start:in_end, :])
             y.append(data[in_end:out_end, 0])
          # move along one time step
          in_start += 1
       return array(X), array(y)

    # train the model
    def build_model(self,train, n_input):
        # prepare data
        train_x, train_y = p1.to_supervised(train, n_input)
        # define parameters
        verbose, epochs, batch_size = 0, 50, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        # reshape output into [samples, timesteps, features]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        # define model
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return model

    # make a forecast
    def forecast(self,model, history, n_input):
        # flatten data
        data = array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
        # retrieve last observations for input data
        input_x = data[-n_input:, :]
        # reshape into [1, n_input, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next week
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    # evaluate a single model
    def evaluate_model(self,train, test, n_input):
        # fit model
        model = p1.build_model(train, n_input)
        # history is a list of weekly data
        history = [x for x in train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(test)):
            # predict the week
            yhat_sequence = p1.forecast(model, history, n_input)
            # store the predictions
            predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test[i, :])
        # evaluate predictions days for each week
        predictions = array(predictions)
        print ("Preds : ", predictions[0])
        self.preds = predictions[0]
        score, scores = p1.evaluate_forecasts(test[:, :, 0], predictions)
        return score, scores


# --- Main  ---
if __name__ == '__main__':
    print("Start Forecast - LSTM by Steps ")

    p1 = Forecast("param_data.json")

    # load the new file
    dataset = p1.getTS()
    # dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
    # print (type(dataset.values),dataset.values)
    # print (type(dataset.values),dataset.index , "  " , np.shape(dataset))
    # exit(0)
    # split into train and test
    train, test = p1.split_dataset(dataset.values)
    # evaluate model and get scores
    n_input = 14
    score, scores = p1.evaluate_model(train, test, n_input)
    # summarize scores
    p1.summarize_scores('lstm', score, scores)
    print (score,"--", len(scores), "-- ", scores,p1.preds)
    for i in range(len(scores)) :
            print (p1.preds[i],scores[i])
            print ("Min : ", p1.preds[i] - scores[i])
            print ("Max : ", p1.preds[i] + scores[i])
            pred_rec = {"Pred": p1.preds[i],"score" : scores[i], "Min" : p1.preds[i] - scores[i], "Max" : p1.preds[i] + scores[i]}
            p1.fullPredsMany.append(pred_rec)
    print (p1.fullPredsMany)


    try:
           response = helpers.bulk(p1.es,p1.fullPredsMany, index='preds')
           print ("\nRESPONSE:", response)
    except Exception as e:
           print("\nERROR:", e)

    print("End of Forecast - LSTM by Steps ")
