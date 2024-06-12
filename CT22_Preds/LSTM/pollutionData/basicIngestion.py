import pandas as pd
from pandas import Series
import time
from datetime import datetime
import json
from elasticsearch import Elasticsearch, helpers
import sys
import numpy as np


class BasicIngestion :

    def __init__(self, param_jason):
       print ("init")
       # self.es = Elasticsearch()
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.path = self.param_data["path"]
       self.pathlength = len(self.path)
       self.pathProcessed= self.param_data["pathProcessed"]
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       # es = Elasticsearch(['http://localhost:9200'], http_auth=('user', 'pass'))
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")


    def read_file_tbi(self) :
       # file_short = input("What file ?:")
       file_short = sys.argv[1]
       file_long = self.path + file_short

       print (file_long)
       # df = read_csv(file_long, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=p1.parser).to_frame()
       # df = read_csv(file_long, header=0, parse_dates=[0], date_parser=p1.parser)
       df = pd.read_csv(file_long)

       print ( "DF : ", df.info())
       dd = pd.concat([df['DEWP'], df['PRES'],df['TEMP']],axis=1)
       print ( "DF : ", dd.describe())
       exit(0)
       timeStamp = pd.concat([df['year'],df['month'],df['day'],df['hour']], axis=1)
       df['timestamp'] = pd.to_datetime(timeStamp)
       # print(df['timestamp'])
       # print (df)
       df['pm2.5'] = df['pm2.5'].fillna(0)
       return(df)

    def ingest(self, df):
      print ("In ingest")
      df_json=df.to_json(orient='records', date_format = 'iso')
      # df_json=df.to_json()
      parsed=json.loads(df_json)

      try:
         response = helpers.bulk(self.es,parsed, index=sys.argv[2])
         # print ("ES response : ", response )
         # print (parsed)
      except Exception as e:
         print ("ES Error :", e)

      return(len(parsed))

# --- Main  ---
if __name__ == '__main__':
    print("Start Basic Ingestion - Pollution Data")

    p1 = BasicIngestion("param_data.json")

    df = p1.read_file_tbi()

    # print (df)
    p1.ingest(df)

    print("End of  Basic Ingestion - Pollution Data")
