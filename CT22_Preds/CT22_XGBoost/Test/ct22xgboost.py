import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import icecream as ic

class ct22xgboost :

   def __init__(self):
       print ("init")


   def mainProcess(self):
       df = pd.read_csv('PJME_hourly.csv')
       df = df.set_index('Datetime')
       df.index = pd.to_datetime(df.index)

       df = df.query('PJME_MW > 19_000').copy()

       train = df.loc[df.index < '01-01-2015']
       test = df.loc[df.index >= '01-01-2015']

       tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)

       df = df.sort_index()

       # df = create_features(df)

       df = self.add_lags(df)

       df = self.cross_validation(df)
       print(df)
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

   def cross_validation(self,df):

       tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
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

    p1 = ct22xgboost()

    p1.mainProcess()

