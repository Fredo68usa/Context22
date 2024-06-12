import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from icecream import ic
from elasticsearch import Elasticsearch, helpers
import sys
import json

class xgboost:



  # --- Constructor ------
  def __init__(self,param_json):

    # self.process = psutil.Process(os.getpid())
    self.timeSeries = str(sys.argv[1])

    # --- Getting parameters from Param File
    with open(param_json) as f:
         self.param_data = json.load(f)

    self.path = self.param_data["path"]
    self.pathlength=len(self.path)
    self.pathProcessed = self.param_data["pathProcessed"]
    self.confidentialityPolicyRule = self.param_data["confidentialityPolicyRule"]
    self.datafileNotinserted=self.pathProcessed + "NotInserted"

    self.esServer = self.param_data["ESServer"]
    self.esUser = self.param_data["ESUser"]
    self.esPwd = self.param_data["ESPwd"]

    self.ExcessiveExtractionCheck = self.param_data["ExcessiveExtractionCheck"]

    self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))

    self.index = self.param_data["index"]
    self.sqlite = self.param_data["sqlite"]
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))

    return(None)
      
      
      
  # --- Read the TS

  # def getTS(self):
  def getTS(self):
        print ("Getting the Time Series")
        # body={"query" : { "bool" : { "must" : [{"match": {"HashHash" : self.SQLHash}} ,{ "match": {"DayOfWeek" : "Thursday"} } ] } } , "size" : 1000}
        body={"query": {"match_all": {}}, "size" : 10000}
        ts_ready_tmp = self.es.search(index=self.timeSeries, body=body)
        # ts_ready_tmp = self.es.search(index='power_consumption_days', body=body)
        data = (ts_ready_tmp['hits']['hits'])
        # myTS_data = pd.DataFrame()
        ic(type(data))
        ic(len(data))
        ic(data)



  def mainProcess(self):
        print ("Starting xgboost")
        # ---- Get the Time Series
        self.getTS()

