# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import json
import psutil
from elasticsearch import Elasticsearch, helpers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class lstm :

    def __init__(self, param_jason):
       # print ("init")
       # self.es = Elasticsearch()
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.path = self.param_data["path"]
       self.pathlength = len(self.path)
       self.pathProcessed= self.param_data["pathProcessed"]
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       self.esIndex = self.param_data["ESIndex"]
       # es = Elasticsearch(['http://localhost:9200'], http_auth=('user', 'pass'))
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       # print ("After connection to ES")

    # ---- getting Collectors MetaData
    def MemConsumption(self):
        # Getting % usage of virtual_memory ( 3rd field)
        print('RAM memory % used:', psutil.virtual_memory()[2])
        # Getting usage of virtual_memory in GB ( 4th field)
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

    def CPUConsumption(self):
        # Calling psutil.cpu_precent() for 4 seconds
        print('The CPU usage is: ', psutil.cpu_percent(4))

    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def insert_many_Elastic(self,newPredictionSet):
      try:

           response = helpers.bulk(self.es,newPredictionSet, index=self.esIndex + '-20221116')
           print ("\nRESPONSE:", response)
           return(response)
      except Exception as e:
           print("\nERROR:", e)


    def TSData(self):
        # print ("TSData")
        p1.CPUConsumption()
        p1.MemConsumption()
        TS_airline_tmp = p1.es.search(index=self.esIndex, body={"query": {"match_all" : {}}, "size" : 10000})
        print (" # 1 : " , type(TS_airline_tmp))
        # for key in TS_airline_tmp['hits']['hits'] :
            # print ("key : ", key["_source"]['Month'])
        # exit(0)
        p1.MemConsumption()

        myArrayTSairline = []
        for hit in TS_airline_tmp['hits']['hits']:
            myArrayTSairline.append(hit["_source"]['Thousands of Passengers'])
            # print (hit["_source"]['Month'], "--" ,hit["_source"]['Thousands of Passengers'])

        dataset = np.array(myArrayTSairline)
        # print (" Its type " , type(dataset))
        # print (" Array " , dataset)
        # exit(0)
        dataset = dataset.reshape(-1, 1)
        dataset = dataset.astype('float32')
        # print (" My array" ,dataset)
        # print (" Its type " , type(dataset))
        # exit(0)

        # Generation of Predictions
        trainPredictPlot, testPredictPlot = p1.TSPreds(dataset)


        # print (testPredictPlot)
        print (len(testPredictPlot))
        # print (trainPredictPlot)
        print (len(trainPredictPlot))

        # for predict in testPredictPlot:
        #for i in range(0,len(testPredictPlot)):
           # print (i,"---",trainPredictPlot[i],testPredictPlot[i])
           # print (i,"---",TS_airline_tmp[i])

        # exit(0)

        docS = []
        i = 0
        for key in TS_airline_tmp['hits']['hits'] :
            # print ("key : ", key["_source"])
            doc= key["_source"]
            # print (i,"---",trainPredictPlot[i][0],testPredictPlot[i][0], doc["Thousands of Passengers"])
            if math.isnan(testPredictPlot[i][0]) == False:
               doc['Pred']=testPredictPlot[i][0]
               # print('Pred test ',testPredictPlot[i][0])
               # print("doc : ",doc)
               docS.append(doc)
            elif math.isnan(trainPredictPlot[i][0]) == False :
               doc['Pred']=trainPredictPlot[i][0]
               # print('Pred train ',testPredictPlot[i][0])
               docS.append(doc)
            else:
               docS.append(doc)


            i = i + 1
            # print (" i : " , i, " -- doc -- ", doc )

        # exit(0)
        response = p1.insert_many_Elastic(docS)
        # print ("Response After call : ",response)


    def TSPreds(self, dataset):

       # fix random seed for reproducibility
       tf.random.set_seed(7)
       # normalize the dataset
       scaler = MinMaxScaler(feature_range=(0, 1))
       dataset = scaler.fit_transform(dataset)
       # split into train and test sets
       train_size = int(len(dataset) * 0.67)
       test_size = len(dataset) - train_size
       train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
       # reshape into X=t and Y=t+3 - (Diff with Regr)
       look_back = 3
       trainX, trainY = p1.create_dataset(train, look_back)
       testX, testY = p1.create_dataset(test, look_back)
       # reshape input to be [samples, time steps, features]
       # Indication of the Steps Here ..
       trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
       testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
       # create and fit the LSTM network
       model = Sequential()
       # model.add(LSTM(4, input_shape=(1, look_back))) - No Steps -
       model.add(LSTM(4, input_shape=(look_back, 1)))
       model.add(Dense(1))
       model.compile(loss='mean_squared_error', optimizer='adam')
       model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
       # make predictions
       trainPredict = model.predict(trainX)
       testPredict = model.predict(testX)
       # invert predictions
       trainPredict = scaler.inverse_transform(trainPredict)
       trainY = scaler.inverse_transform([trainY])
       testPredict = scaler.inverse_transform(testPredict)
       testY = scaler.inverse_transform([testY])
       # calculate root mean squared error
       trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
       print('Train Score: %.2f RMSE' % (trainScore))
       testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
       print('Test Score: %.2f RMSE' % (testScore))
       # shift train predictions for plotting
       trainPredictPlot = np.empty_like(dataset)
       trainPredictPlot[:, :] = np.nan
       trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
       # shift test predictions for plotting
       testPredictPlot = np.empty_like(dataset)
       testPredictPlot[:, :] = np.nan
       testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
       # plot baseline and predictions
       # plt.plot(scaler.inverse_transform(dataset))
       # plt.plot(trainPredictPlot)
       # plt.plot(testPredictPlot)
       # plt.show()
       return ( trainPredictPlot, testPredictPlot )
 

# --- Main  ---
if __name__ == '__main__':
    print("Start LSTM airline")

    p1 = lstm("param_data.json")

    p1.TSData()

    print("End of LSTM airline")
