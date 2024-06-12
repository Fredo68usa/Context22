from datetime import datetime
from elasticsearch import Elasticsearch, helpers
import pandas as pd
import numpy as np
import csv
import urllib.parse
import json
import getpass
import datetime as dt
import time

class TSGen :

  def __init__(self,param_json):

    # self.es = Elasticsearch()

    # self.process = psutil.Process(os.getpid())
    # self.collector = str(sys.argv[1])

    with open(param_json) as f:
         self.param_data = json.load(f)

    self.esServer = self.param_data["ESServer"]
    self.esUser = self.param_data["ESUser"]
    self.esPwd = self.param_data["ESPwd"]

    self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))

    # self.index = self.param_data["index"]
    self.myFSQLts = [] 
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    # print('Argument List:', str(sys.argv))

    # --- Read off Full SQL

    # ---- getting the TS
  def getTS(self):
        print ("Getting the Time Series")

        body={"query": {"match_all" : {}}, "size" : 10000}

        ts_FullSQL_tmp = self.es.search(index="enriched_full_sql", body=body)
        ts_FullSQL_tmp = self.es.search(index="enriched_full_sql", body={"query": {"match_all" : {}}, "size" : 10000})

        for hit in ts_FullSQL_tmp['hits']['hits']:
           RecAff= hit["_source"]['Records Affected']
           ts_tmp= hit["_source"]['Timestamp Local Time']
           TSLocal=dt.datetime.strptime(ts_tmp[:10],'%Y-%m-%d')
           DoY = hit["_source"]['DayOfYear']
           DoW = hit["_source"]['DayOfWeek']
           Year = hit["_source"]['Year']
           Hash  = hit["_source"]['HashHash']

           self.myFSQLts.append([Hash,RecAff,TSLocal,Year,DoY,DoW])

        df = pd.DataFrame (self.myFSQLts , columns = ['HashHash','Records Affected','Timestamp Local Time','Year','DayOfYear','DayOfWeek'])
        print(df)
        table = pd.pivot_table(df, index=['HashHash', 'Year', 'DayOfYear','DayOfWeek'], values=['Records Affected', 'Timestamp Local Time'], aggfunc={'Records Affected': np.sum, 'Timestamp Local Time': np.min, 'Timestamp Local Time': np.max})
        print (table.reset_index())
# --- Aggregate via Pivot table


# --- Writing into postgresql

  def mainProcess(self):
        self.getTS()
