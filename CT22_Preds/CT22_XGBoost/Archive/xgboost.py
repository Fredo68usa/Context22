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
    # self.timeSeries = str(sys.argv[1])
    self.timeSeries = 'power_consumption_days'

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
        # body={"query": {"match_all": {}}, "size" : 10000}
        fields_to_retrieve = ['timestamp','Global_active_power']
        body={"query": {"match_all": {}}, "size" : 10000, "_source": fields_to_retrieve}
        ts_ready_tmp = self.es.search(index=self.timeSeries, body=body)
        # ts_ready_tmp = self.es.search(index='power_consumption_days', body=body)
        # data = (ts_ready_tmp['hits']['hits'])

        # data = pd.DataFrame(ts_ready_tmp['hits']['hits'])
        data = ts_ready_tmp['hits']['hits']
        data = pd.DataFrame.from_dict(ts_ready_tmp['hits']['hits'])
        # data = pd.DataFrame.from_dict(ts_ready_tmp['hits']['hits']['_source'])

        # ic(data['timestamp'])
        # jsonData = json.loads(data[0])
        # ic(jsonData)
        # myTS_data = pd.DataFrame()
        # ic(type(data))
        # ic(len(data))

        list_ts = data['_source'].tolist()
        ic(list_ts)
        self.df_ts = pd.DataFrame.from_dict(list_ts)
        self.df_ts['timestamp'] = pd.to_datetime(self.df_ts['timestamp'])
        ic(self.df_ts)

  def trainTS(self):
      # --- train/test = 2/3
      index_train = 1442 * (2/3)

      self.train = self.df_ts.loc[self.df_ts.index < index_train]
      self.test = self.df_ts.loc[self.df_ts.index >= index_train]

      ic(self.test)

  def create_features(self):
    """
    Create time series features based on time series index.
    """
    ic(self.df_ts)
    ic(type(self.df_ts))
    self.df_ts=self.df_ts.set_index('timestamp')
    ic(self.df_ts)
    # exit(0)
    # self.df_ts = self.df_ts.copy()
    # self.df_ts['hour'] = self.df_ts['timestamp'].hour
    self.df_ts['hour'] = self.df_ts.index.hour
    # self.df_ts['hour'] = self.df_ts['timestamp'].timetuple()
    # df['dayofweek'] = df.index.dayofweek
    # df['quarter'] = df.index.quarter
    # df['month'] = df.index.month
    # df['year'] = df.index.year
    # df['dayofyear'] = df.index.dayofyear
    # df['dayofmonth'] = df.index.day
    # df['weekofyear'] = df.index.isocalendar().week

    ic(self.df_ts)


  def mainProcess(self):
    print ("Starting xgboost")
    # ---- Get the Time Series
    self.getTS()

    self.trainTS()

    self.create_features()
