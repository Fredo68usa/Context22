from datetime import datetime
from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np
import csv
# import urllib.parse
import json
# import getpass
import datetime as dt
import time


# ---------------------------
#    Version 1.1 : Taking into account 
#        def mainProcess(self):
# ---------------------------
class TSGen :

  def __init__(self,param_json):


    with open(param_json) as f:
         self.param_data = json.load(f)

    self.esServer = self.param_data["ESServer"]
    self.esUser = self.param_data["ESUser"]
    self.esPwd = self.param_data["ESPwd"]
    self.SQLHash = self.param_data["hash"]
    # self.Year = self.param_data["Year"]
    # self.esTZ = self.param_data["ESTZ"]
    self.esInIndex = self.param_data["ESInIndex"]
    self.esOutIndex = self.param_data["ESOutIndex"]

    self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))



  def getTS(self, hsh):
    print('In getTS')
    if (not hsh) :
       hsh = self.SQLHash
    # fields_to_retrieve = ['timestamp','Global_active_power']
    fields_to_retrieve = ['Year','DayOfYear','HashHash','data.recordsaffected','Records Affected','DayOfWeek']
    # body={"query": {"match_all": {}}, "size" : 10000, "_source": fields_to_retrieve}
    
    body={"query": 
            {"match": 
              {
                "HashHash" : hsh
              }
            }, 
            "size" : 10000, "_source": fields_to_retrieve
            }

    ts_ready_tmp = self.es.search(index=self.esInIndex, body=body)
    # breakpoint()
    data0 = ts_ready_tmp['hits']['hits']
    if not data0:
        print ("No Data for that SQL : " , hsh )
        return()
    # data = pd.DataFrame.from_dict(ts_ready_tmp['hits']['hits'])
    data = pd.DataFrame.from_dict(data0)
    data2=data['_source'].tolist()
    data3=pd.DataFrame.from_dict(data2)
    breakpoint()

    return(data3,ts)

  def pivotTS (self):
    # --- Aggregate via Pivot table
        df = pd.DataFrame (self.ts , columns = ['HashHash','Records Affected','Timestamp Local Time','Year','DayOfYear','DayOfWeek'])
        # print(df)
        print ("PIVOT")
        table = pd.pivot_table(df, index=['HashHash', 'Year', 'DayOfYear','DayOfWeek'], 
                values=['Records Affected', 'Timestamp Local Time'], 
                # aggfunc={'Records Affected': np.sum, 'Timestamp Local Time': np.min, 'Timestamp Local Time': np.max})
                aggfunc={'Records Affected': np.sum})
        self.df2 = table.reset_index()
        # print (self.df2)
       
        # --- Taking into account the Validated Anomalies














  def mainProcess(self, hsh):
      ts = self.getTS(hsh)
      ts_day = self.pivotTS()
        # self.putInES()

