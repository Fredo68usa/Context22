import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from icecream import ic
import json
from elasticsearch import Elasticsearch, helpers
import sys


class ct22xgboost :

   def __init__(self,param_json):
    print ("init")

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
        fields_to_retrieve = ['timestamp','Global_active_power']
        body={"query": {"match_all": {}}, "size" : 10000, "_source": fields_to_retrieve}
        ts_ready_tmp = self.es.search(index=self.timeSeries, body=body)

        data = ts_ready_tmp['hits']['hits']
        data = pd.DataFrame.from_dict(ts_ready_tmp['hits']['hits'])

        list_ts = data['_source'].tolist()
        ic(list_ts)
        self.df_ts = pd.DataFrame.from_dict(list_ts)
        self.df_ts['timestamp'] = pd.to_datetime(self.df_ts['timestamp'])
        ic(self.df_ts)


   def getCsv(self):
       df = pd.read_csv('PJME_hourly.csv')
       df = df.set_index('Datetime')
       df.index = pd.to_datetime(df.index)

       df = df.query('PJME_MW > 19_000').copy()

       train = df.loc[df.index < '01-01-2015']
       test = df.loc[df.index >= '01-01-2015']
       return(df)

   def mainProcess(self):

       df = self.getCsv()

       # cont = True
       # while cont==True:
       #     splits = int(input("Nbr of Splits  5 "))
       #     size = int(input("Test Size : 24*365*1 = 8760 "))
       #     gap = int(input("Nbr of Gaps : 24 "))
       #     # splits = int(input("Nbr of Splits"))
       #     # tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
       #     tss = TimeSeriesSplit(n_splits=splits, test_size=size, gap=gap)
       #     ic(tss)
       #     cont_in = input ("Again  y/n  : ")
       #     if cont_in == "y" :
       #         cont = True
       #     else :
       #         cont = False
       # exit(0)

       df = df.sort_index()

       df = self.add_lags(df)

       # df = self.cross_validation(df)

       cont = True
       while cont==True:
           splits = int(input("Nbr of Splits  5 "))
           size = int(input("Test Size : 24*365*1 = 8760 "))
           gap = int(input("Nbr of Gaps : 24 "))
           # splits = int(input("Nbr of Splits"))
           # tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
           tss = TimeSeriesSplit(n_splits=splits, test_size=size, gap=gap)
           ic(tss)
           df = self.cross_validation(df,tss)
           # ic(tss)
           cont_in = input ("Again  y/n  : ")
           if cont_in == "y" :
               cont = True
           else :
               cont = False

       # exit(0)
       # print(df)
       df = self.predict_future(df)

   def create_features(self,df):
       """
       Create time series features based on time series index.
       """
       df = df.copy()
       df['hour'] = df.index.hour
       df['dayofweek'] = df.index.dayofweek
       df['quarter'] = df.index.quarter
       df['month'] = df.index.month
       df['year'] = df.index.year
       df['dayofyear'] = df.index.dayofyear
       df['dayofmonth'] = df.index.day
       df['weekofyear'] = df.index.isocalendar().week
       return df

   def add_lags(self,df):
       target_map = df['PJME_MW'].to_dict()
       df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
       df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
       df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
       return df

   def cross_validation(self,df,tss):

       # tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
       df = df.sort_index()
       fold = 0
       preds = []
       scores = []
       for train_idx, val_idx in tss.split(df):
           train = df.iloc[train_idx]
           test = df.iloc[val_idx]
           train = self.create_features(train)
           test = self.create_features(test)
           FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year','lag1','lag2','lag3']
           TARGET = 'PJME_MW'
           X_train = train[FEATURES]
           y_train = train[TARGET]
           X_test = test[FEATURES]
           y_test = test[TARGET]
           reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
       reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
       y_pred = reg.predict(X_test)
       preds.append(y_pred)
       score = np.sqrt(mean_squared_error(y_test, y_pred))
       scores.append(score)
       return(df)

   def predict_future(self,df):
       #  Retrain on all data
       print("predict_future")
       df = self.create_features(df)

       FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'lag1','lag2','lag3']
       TARGET = 'PJME_MW'

       X_all = df[FEATURES]
       y_all = df[TARGET]

       reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
       reg.fit(X_all, y_all,
           eval_set=[(X_all, y_all)],
            verbose=100)

       print (df.index.max())

       # Create future dataframe
       future = pd.date_range('2018-08-03','2019-08-01', freq='1h')
       future_df = pd.DataFrame(index=future)
       future_df['isFuture'] = True
       df['isFuture'] = False
       df_and_future = pd.concat([df, future_df])
       df_and_future = self.create_features(df_and_future)
       df_and_future = self.add_lags(df_and_future)

       future_w_features = df_and_future.query('isFuture').copy()

       future_w_features['pred'] = reg.predict(future_w_features[FEATURES])

       print (future_w_features)

       # --- Main  ---
if __name__ == '__main__':
    print("Start BUFF Enrichment")

    p1 = ct22xgboost("param_data.json")

    p1.mainProcess()

