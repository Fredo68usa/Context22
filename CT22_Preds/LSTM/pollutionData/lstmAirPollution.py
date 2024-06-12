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

        # declare a filter query dict object
        match_all = {
            "size": 100,
            "query": {
                "match_all": {}
            }
           }
        # make a search() request to get all docs in the index
        resp = p1.es.search(
            index = self.esIndex,
            body = match_all,
            scroll = '2s' # length of time to keep search context
            )

        # keep track of pass scroll _id
        old_scroll_id = resp['_scroll_id']

        # keep track of the number of the documents returned
        doc_count = 0

        print ("old_scroll_id : " , old_scroll_id )

        # Declare a python LIST
        myArrayTSpollution = []

        # use a 'while' iterator to loop over document 'hits'
        while len(resp['hits']['hits']):
            # make a request using the Scroll API
            resp = p1.es.scroll(
                scroll_id = old_scroll_id,
                scroll = '2s' # length of time to keep search context
            )

            # check if there's a new scroll ID
            if old_scroll_id != resp['_scroll_id']:
                print ("NEW SCROLL ID:", resp['_scroll_id'])

            # keep track of pass scroll _id
            old_scroll_id = resp['_scroll_id']

            # print the response results
            # print ("\nresponse for index:", self.esIndex)
            # print ("_scroll_id:", resp['_scroll_id'])
            # print ('response["hits"]["total"]["value"]:', resp["hits"]["total"]["value"])

            myArrayTSpollution_line = []

            # iterate over the document hits for each 'scroll'
            for doc in resp['hits']['hits']:
                # print ("\n", doc['_id'], doc['_source'])
                # print(doc["_source"]['TEMP'])
                TSpollution_line = [doc["_source"]['DEWP'],doc["_source"]['TEMP'],doc["_source"]['PRES'],doc["_source"]['pm2.5']]
                myArrayTSpollution_line.append(TSpollution_line)
                print("Type of doc : " , type(doc))
                print(myArrayTSpollution_line)
                doc_count += 1
                # print ("DOC COUNT:", doc_count)

            myArrayTSpollution.append(myArrayTSpollution_line)

        # print the total time and document count at the end
        print (myArrayTSpollution)
        print ("\nTOTAL DOC COUNT:", doc_count)
        print ("\n Length Of Array:", len(myArrayTSpollution))

        p1.MemConsumption()

        exit(0)

        dataset = np.array(myArrayTSpollution)
        dataset = dataset.reshape(-1, 1)
        dataset = dataset.astype('float32')

        # Generation of Predictions
        trainPredictPlot, testPredictPlot = p1.TSPreds(dataset)

        print (len(testPredictPlot))
        print (len(trainPredictPlot))

        docS = []
        i = 0
        for key in TS_airline_tmp['hits']['hits'] :
            doc= key["_source"]
            if math.isnan(testPredictPlot[i][0]) == False:
               doc['Pred']=testPredictPlot[i][0]
               docS.append(doc)
            elif math.isnan(trainPredictPlot[i][0]) == False :
               doc['Pred']=trainPredictPlot[i][0]
               docS.append(doc)
            else:
               docS.append(doc)


            i = i + 1

        response = p1.insert_many_Elastic(docS)


    def TSPreds(self, dataset):

       return ()
 

# --- Main  ---
if __name__ == '__main__':
    print("Start LSTM airline")

    p1 = lstm("param_data.json")

    p1.TSData()

    print("End of LSTM airline")
