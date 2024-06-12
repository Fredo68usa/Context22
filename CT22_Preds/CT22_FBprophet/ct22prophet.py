# Python
import pandas as pd
from prophet import Prophet
from elasticsearch import Elasticsearch, helpers
import json
from datetime import datetime


class ct22fbprophet:

    def __init__(self, param_jason):
       print ("init")
       # self.es = Elasticsearch()
       with open(param_jason) as f:
          self.param_data = json.load(f)

       self.path = self.param_data["path"]
       self.pathlength = len(self.path)
       self.pathProcessed= self.param_data["pathProcessed"]
       self.index = self.param_data["index"]
       self.esServer = self.param_data["ESServer"]
       self.esUser = self.param_data["ESUser"]
       self.esPwd = self.param_data["ESPwd"]
       self.sqlite = self.param_data["sqlite"]
       # es = Elasticsearch(['http://localhost:9200'], http_auth=('user', 'pass'))
       self.es = Elasticsearch([self.esServer], http_auth=(self.esUser, self.esPwd))
       print ("After connection to ES")




    def into_ES(self, parsed):
      # breakpoint()
      try:
         response = helpers.bulk(self.es,parsed, index='ct22_fb_prophet')
         print ("ES response : ", response )
      except Exception as e:
         print ("ES Error :", e)
         pass

      return(len(parsed))

    def prophet_prog(self) :

        # Python
        # df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
        df = pd.read_csv('../CT22_XGBoost/Test/PJME_hourly.csv')

        df.rename(columns={ 'Datetime': 'ds', 'PJME_MW': 'y'} ,inplace = True)
        print(list(df.columns))
        print(df.head())


        m = Prophet()
        m.fit(df)

        print('Fit done')

        # future = m.make_future_dataframe(periods=365)
        future = m.make_future_dataframe(periods=3)
        print(future.tail())


        forecast = m.predict(future)
        # forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
        # breakpoint()
        print(forecast)
        ct22_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        ct22_forecast['Series']= 'PJME_MW'
        # ct22_forecast['date'] = pd.to_datetime(ct22_forecast['ds'])
        # ct22_forecast['timestamp'] = ct22_forecast['date'].astype('int64') // 10**9
        ct22_forecast_es = ct22_forecast.to_dict(orient='records') 
        # breakpoint()
        count = self.into_ES(ct22_forecast_es)


# --- Main  ---
if __name__ == '__main__':
    print("Start CT22 FB Prophet")

    p1 = ct22fbprophet("param_data.json")

    p1.prophet_prog()


    print("End CT22 FB Prophet")



