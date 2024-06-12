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
import ct22posgresql as psg
# ---------------------------
#    Version 1.1 : Taking into account 
#         the anomalies
# ---------------------------
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
    self.SQLHash = self.param_data["hash"]
    self.Year = self.param_data["Year"]
    # self.esTZ = self.param_data["ESTZ"]
    # self.esInIndex = self.param_data["ESInIndex"]
    # self.esOutIndex = self.param_data["ESOutIndex"]

    self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))

    # self.index = self.param_data["index"]
    self.myFSQLts = [] 
    self.df2 = None
    # print('Number of arguments:', len(sys.argv), 'arguments.')
    # print('Argument List:', str(sys.argv))

    # ---- Opening PostGreSQL
    p1 = psg.CT22PosGreSQL()
    p1.open_PostGres()

    print(p1.postgres_connect)
    print (" Read PosGres" )
    body = """SELECT * from currentpreds where hash=%s and year=%s"""
    # self.cursor.execute("SELECT version();")
    # cur.execute(sql, (value1,value2))
    p1.cursor.execute(body,(self.SQLHash,self.Year))
    records = p1.cursor.fetchall()
    print("Read Predictions - ", records,"\n")
    # print("Read Predictions - ","\n")
    self.preds_df = pd.DataFrame(records,columns=['hash','year','dayofyear','predstype','preds','preds_interval','anomaly','excessqty'])
    # print(self.preds_df)
    # exit(0)


    # ---- getting the TS
  def getTS(self):
        print ("Getting the Time Series")

        # body={"query": {"match_all" : {}}, "size" : 10000}
        body={"query": {"match_all" : {}}, "size" : 10000}

        ts_FullSQL_tmp = self.es.search(index="enriched_full_sql", body=body)

        for hit in ts_FullSQL_tmp['hits']['hits']:
           RecAff= hit["_source"]['Records Affected']
           ts_tmp= hit["_source"]['Timestamp Local Time']
           TSLocal=dt.datetime.strptime(ts_tmp[:10],'%Y-%m-%d')
           DoY = hit["_source"]['DayOfYear']
           DoW = hit["_source"]['DayOfWeek']
           Year = hit["_source"]['Year']
           Hash  = hit["_source"]['HashHash']

           self.myFSQLts.append([Hash,RecAff,TSLocal,Year,DoY,DoW])

  def pivotTS (self):
    # --- Aggregate via Pivot table
        df = pd.DataFrame (self.myFSQLts , columns = ['HashHash','Records Affected','Timestamp Local Time','Year','DayOfYear','DayOfWeek'])
        # print(df)
        print ("PIVOT")
        table = pd.pivot_table(df, index=['HashHash', 'Year', 'DayOfYear','DayOfWeek'], values=['Records Affected', 'Timestamp Local Time'], aggfunc={'Records Affected': np.sum, 'Timestamp Local Time': np.min, 'Timestamp Local Time': np.max})
        self.df2 = table.reset_index()
        # print (self.df2)
       
        # --- Taking into account the Anomalies
        for i in range(len(self.df2)):
            predFoundDf = self.preds_df[(self.preds_df.hash == self.df2.iloc[i]['HashHash']) & (self.preds_df.year == self.df2.iloc[i]['Year'])& (self.preds_df.dayofyear == self.df2.iloc[i]['DayOfYear'])]
            if predFoundDf.empty == False :
               print (predFoundDf['excessqty'])
               print (self.df2.loc[i]['Records Affected'])
               self.df2.at[i,'Records Affected'] = self.df2.loc[i]['Records Affected'] - predFoundDf['excessqty']
               print (self.df2.loc[i]['Records Affected'])

        # exit(0)


# --- Writing into ES
  def putInES(self) :
      df_json=self.df2.to_json(orient='records', date_format = 'iso')
      # df_json=df.to_json()
      parsed=json.loads(df_json)
      # print(parsed)
      # exit(0)

      try:
         # response = helpers.bulk(self.es,parsed, index=sys.argv[2])
         response = helpers.bulk(self.es,parsed, index='ts_ready_2')
         print ("ES response : ", response )
         # print (parsed)
      except Exception as e:
         print ("ES Error :", e)

  def mainProcess(self):
        self.getTS()
        self.pivotTS()
        # self.putInES()

