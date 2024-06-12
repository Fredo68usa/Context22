import pandas as pd
import numpy as np
import json
import datetime as dt
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters 
# single exponential smoothing
# double and triple exponential smoothing
from sklearn.metrics import mean_absolute_error,mean_squared_error
from elasticsearch import Elasticsearch, helpers
import ct22posgresql as psg
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
import tensorflow as tf 


class TSPredGenLSTM:

    def __init__(self, param_jason):
       # print ("init")
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.myTS_data = pd.DataFrame()
       self.myTS = pd.DataFrame()
       self.myTS_train = pd.DataFrame()
       self.myTS_fitted = pd.DataFrame()
       self.myTS_global = pd.DataFrame()
       self.preds_df = pd.DataFrame()
       self.alpha = float()
       self.SQLHash = self.param_data["SQLHash"]
       self.DayOfWeek = self.param_data["DayOfWeek"]
       self.esTZ = self.param_data["ESTZ"]
       self.esInIndex = self.param_data["ESInIndex"]
       self.esOutIndex = self.param_data["ESOutIndex"]
       self.SeasonalPeriod = self.param_data["SeasonalPeriod"]
       self.Alpha = self.param_data["Alpha"]
       self.IndexFreq = self.param_data["IndexFreq"]
       self.Train = self.param_data["Train"]
       self.Forecast = self.param_data["Forecast"]

       self.now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       # print (self.now , type(self.now))

    def getPreds(self,p1):
       # body = """SELECT * from currentpreds where hash='%s';"""
       body = """SELECT * from currentpreds;"""
       # self.cursor.execute("SELECT version();")
       # cur.execute(sql, (value1,value2))
       # p1.cursor.execute(body,self.SQLHash)
       p1.cursor.execute(body)
       records = p1.cursor.fetchall()
       # print("Read Predictions - ", records,"\n")
       # print("Read Predictions - ","\n")
       self.preds_df = pd.DataFrame(records,columns=['hash','year','dayofyear','predstype','preds','preds_interval','anomaly','excessqty'])
       # print(self.preds_df)

       # exit(0)

    def elkOpen(self):
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")

    # ---- getting the TS
    def getTS(self):
        print ("Getting the Time Series")
        body={"query" : { "bool" : { "must" : [{"match": {"HashHash" : self.SQLHash}} ,{ "match": {"DayOfWeek" : "Thursday"} } ] } } , "size" : 1000}
        ts_ready_tmp = self.es.search(index=self.esInIndex, body=body)
        data = (ts_ready_tmp['hits']['hits'])
        # myTS_data = pd.DataFrame()
        for row in range(len(data)) :
            src =  data[row]['_source']
            self.myTS_data=self.myTS_data.append(pd.json_normalize(src))
            # self.myTS_data['Time of Run'] = self.now
            # print(self.myTS_data)
        # myTS_data.reset_index(drop=True)
        # myTS_data.index=myTS_data['HashHash']
        self.myTS_data.reset_index(drop=True,inplace=True)
        # print('self.myTS_data', self.myTS_data)
        # print('self.myTS_data', self.myTS_data)
        # exit(0)

        # --- Taking into account the Anomalies
        for i in range(len(self.myTS_data)):
            predFoundDf = self.preds_df[(self.preds_df.hash == self.myTS_data.iloc[i]['HashHash']) & (self.preds_df.year == self.myTS_data.iloc[i]['Year'])& (self.preds_df.dayofyear == self.myTS_data.iloc[i]['DayOfYear'])]
            if predFoundDf.empty == False :
               # print (predFoundDf['excessqty'])
               # print (self.myTS_data.loc[i]['Records Affected'])
               # print (self.myTS_data[i]['Records Affected'])
               self.myTS_data.at[i,'Records Affected'] = self.myTS_data.loc[i]['Records Affected'] - predFoundDf['excessqty']

        self.myTS_data.rename(columns = {'Timestamp Local Time':'Date', 'Records Affected':'Quantity'}, inplace = True)
        self.myTS_data = self.myTS_data.set_index('Date')
        self.myTS_data = self.myTS_data.sort_index()

        # print(" ---> self.myTS_data")
        # print(self.myTS_data)
        # exit(0)

    def put_TS_Train_Test_Init(self):

      # print(" data : " , self.myTS_data)
      nbrTrainPeriod = int(self.Train*(len(self.myTS_data)))
      self.nbrTestPeriod = len(self.myTS_data) - nbrTrainPeriod
      self.myTS_train = self.myTS_data[:nbrTrainPeriod].copy()
      self.myTS_test = self.myTS_data[nbrTrainPeriod:].copy()
      # print(self.myTS_train)
      # print(self.myTS_test)

      # Prepare for Preds Computation
      self.TS_test_fit = self.myTS_test.copy()
      # print ("===========================================================================")

      self.TS_test_fit.drop(['Quantity'], axis=1,inplace = True )
      # print(self.myTS_test)
      # print(self.TS_test_fit)


      # exit(0)


    def put_TS_Train_Test(self,algo):
      print ( " ============== ")
      print ( " In put_TS_Train_test ")
      print ( " ============== ")

      TS_train = self.myTS_train.copy()
      TS_test = self.myTS_test.copy()
      TS_train["Type Of Data"] = "Train"
      TS_train["Time Of Simul"] = self.now
      TS_train["Algorithm"] = algo
      TS_train["DataTypeAlgo"] = "Train " + algo
      TS_test["Type Of Data"] = "Test"
      TS_test["Time Of Simul"] = self.now
      TS_test["Algorithm"] = algo
      TS_test["DataTypeAlgo"] = "Test " + algo

      print(TS_train)
      print(TS_test)

      # Record Train set
      TS_train.reset_index(inplace=True)
      df_json=TS_train.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      # print (parsed)
      self.insertES_bulk(parsed)

      # record Test set
      TS_test.reset_index(inplace=True)
      df_json=TS_test.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      # print (parsed)
      self.insertES_bulk(parsed)


    def put_Fit(self, algo, TS_fit):
      TS_fit.reset_index(inplace=True)
      # print(TS_fit)
      # exit(0)
      TS_fit.rename(columns = {'index':'Date' , 0 :'Quantity'}, inplace = True)
      TS_fit['Date'] = pd.to_datetime(TS_fit.Date)
      TS_fit['Date'] = TS_fit['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')

      TS_fit.set_index('Date', inplace = True )
      print(TS_fit)

      print("---- In put Fit -----")
      print(type(self.TS_test_fit))
      print(self.TS_test_fit)
      # TS_test_fit.drop(['Quantity'], axis=1,inplace = True )
      # print(TS_test_fit)
      TS_test_fit = pd.concat([self.TS_test_fit,TS_fit[:len(self.myTS_test)]], axis=1)

      print(TS_test_fit)
      # print(TS_test)

      # exit(0)

      # record Test Fit set
      TS_test_fit["Type Of Data"] = "Test Fit"
      TS_test_fit["Time Of Simul"] = self.now
      TS_test_fit["Algorithm"] = algo
      TS_test_fit["DataTypeAlgo"] = "Test Fit " + algo
      print(TS_test_fit)

      # exit(0)

      TS_test_fit.reset_index(inplace=True)
      df_json=TS_test_fit.to_json(orient='records', date_format = 'iso')
      parsed=json.loads(df_json)
      print (parsed)
      self.insertES_bulk(parsed)

      # exit(0)

   
    def dataLSTMProcess(self,rdn):
        # from sklearn.preprocessing import MinMaxScaler
        # np.random.seed(rdn)
        # tf.random.set_seed(rdn)

        scaler = MinMaxScaler()
        # fit the format of the scaler -> convert shape from (1000, ) -> (1000, 1)
        # qty = df.qty.values.reshape(-1, 1)
        qty = self.myTS_data.Quantity.values.reshape(-1, 1)
        scaled_qty = scaler.fit_transform(qty)
        # print(qty)
        # print(scaled_qty)
        # print(scaler.inverse_transform(scaled_qty))

        # seq_len = 60
        seq_len = 4
        # seq_len = input("seq_len : " )
        # seq_len = int(seq_len)
        train_frac=0.9

        x_train, y_train, x_test, y_test = self.get_train_test_sets(scaled_qty, seq_len, train_frac )

        print (" x_train : " , x_train.shape)
        print (" y_train : " , y_train.shape)
        print (" x_test : " , x_test.shape)
        print (" y_test : " , y_test.shape)

        # exit(0)

        # fraction of the input to drop; helps prevent overfitting
        dropout = 0.2
        window_size = seq_len - 1

        # build a 3-layer LSTM RNN
        # print (" Model Generation ")
        model = keras.Sequential()

        # print (" Model Add 1 ")
        model.add(
             LSTM(window_size, return_sequences=True,
                  input_shape=(window_size, x_train.shape[-1]))
             )

        # print (" Model Add 2 ")
        model.add(Dropout(rate=dropout))
        # Bidirectional allows for training of sequence data forwards and backwards
        model.add(
             Bidirectional(LSTM((window_size * 2), return_sequences=True)
             ))

        # print (" Model Add 3 ")
        model.add(Dropout(rate=dropout))
        model.add(
             Bidirectional(LSTM(window_size, return_sequences=False))
             ) 

        # print (" Model Add 4 ")
        model.add(Dense(units=1))
        # linear activation function: activation is proportional to the input
        # print (" Model Add 5 ")
        model.add(Activation('linear'))
        # print (" model type " , type(model))

        # batch_size = 16
        batch_size = 4

        # print (" Model Compile w/ MSE")
        model.compile(
        loss='mean_squared_error',
        optimizer='adam'
        )

        print (" ============ ")
        print (" Model History - en fait fit ")
        print (" ============ ")
        history = model.fit(
        x_train,
        y_train,
        # epochs=10,
        epochs=4,
        batch_size=batch_size,
        shuffle=False,
        validation_split=0.2
        )
        print (" history type " , type(history))
        print (" history " , history.history)

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        results = model.evaluate(x_test, y_test, batch_size=batch_size*2)
        print("test loss, test acc:", results)


        # Generate prediction for 3 samples
        y_pred = model.predict(x_test[:3])
        print("predictions shape:", y_pred.shape)

        # invert the scaler to get the absolute data
        y_test_orig = scaler.inverse_transform(y_test)
        y_pred_orig = scaler.inverse_transform(y_pred)

        print(y_test_orig)
        print(y_pred_orig)


    def split_into_sequences(self,data, seq_len):
        n_seq = len(data) - seq_len + 1
        return np.array([data[i:(i+seq_len)] for i in range(n_seq)])

    def get_train_test_sets(self , data, seq_len, train_frac):
        print (data.shape , " -- " , type(data))
        # exit(0)
        sequences = self.split_into_sequences(data, seq_len)
        # print("sequences : " , sequences)
        n_train = int(sequences.shape[0] * train_frac)
        print("n_train : " , n_train)
        X_train = sequences[:n_train, :-1, :]
        y_train = sequences[:n_train, -1, :]
        X_test = sequences[n_train:, :-1, :]
        y_test = sequences[n_train:, -1, :]
        print (" -- X_train " , X_train.shape , " -- " , type(X_train))
        exit(0)
        return X_train, y_train, X_test, y_test

    def mainProcess(self):

        # Opening the postGresql DB
        p1 = psg.CT22PosGreSQL()
        p1.open_PostGres()

        # Getting the Preds
        self.getPreds(p1)

        # Getting the TS
        self.elkOpen()
        self.getTS()
        # self.frameShape()
        # print("After frameShape")

        # Recording of the TS in ES as Train/Test sets
        self.put_TS_Train_Test_Init()

        # rdn_list = [20, 60, 90, 200, 600, 1000, 1234]
        rdn_list = [20, 60, 1234]
        for rdn in rdn_list:
            print(" rdn : " , rdn)
            self.dataLSTMProcess(rdn)

        # exit(0)
        # Computng the pred interval
        # self.rmse_HWES3MUL, self.preds_interval_HWES3MUL = self.errorComp(self.myTS['Quantity'],self.myTS_fitted['HWES3_MUL'])
        # self.rmse_HWES1 , self.preds_interval_HWES1 = self.errorComp(self.myTS['Quantity'],self.myTS_fitted['HWES1'])


        # Recording the reference pred
        # preType = input (" Select type of pred to record : 1-HWES3MUL ")
        # p1 = psg.CT22PosGreSQL()
        # p1.open_PostGres()

        # print(p1.postgres_connect)
        # print(p1.cursor)
        # self.write_PosGres(p1)
        p1.close_PosGres()


    def insertES_bulk(self,parsed):
      # exit(0)
      try:
         response = helpers.bulk(self.es,parsed, index=self.esOutIndex)
         print ("ES response : ", response )
      except Exception as e:
         print ("ES Error :", e)

