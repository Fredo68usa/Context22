import math
import numpy as np
import pandas as pd
import time

from datetime import date
# from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
# from tqdm import tqdm.notebook.tqdm
from xgboost import XGBRegressor


from icecream import ic
import pdb
import json
from elasticsearch import Elasticsearch, helpers
import sys

"""

https://medium.com/@redeaddiscolll/forecasting-stock-prices-with-xgboost-0b79fdcdd9ae

"""


class ct22xgboost:


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

    # return(None)

    #### Input params ##################
    self.stk_path = "./data/VTI.csv"
    self.test_size = 0.2                # proportion of dataset to be used as test set
    self.cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
    self.N = 3                         # for feature at day t, we use lags from t-1, t-2, ..., t-N as features

    self.n_estimators = 100             # Number of boosted trees to fit. default = 100
    self.max_depth = 3                  # Maximum tree depth for base learners. default = 3
    self.learning_rate = 0.1            # Boosting learning rate (xgb’s “eta”). default = 0.1
    self.min_child_weight = 1           # Minimum sum of instance weight(hessian) needed in a child. default = 1
    self.subsample = 1                  # Subsample ratio of the training instance. default = 1
    self.colsample_bytree = 1           # Subsample ratio of columns when constructing each tree. default = 1
    self.colsample_bylevel = 1          # Subsample ratio of columns for each split, in each level. default = 1
    self.gamma = 0                      # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

    self.model_seed = 100

    self.scaler = []

    fontsize = 14
    ticklabelsize = 14
    ####################################



  def read_data(self):
       # Read Data
       df = pd.read_csv(self.stk_path, sep = ",")

       ic(df.head())

       # Convert Date column to datetime
       # df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
       # df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d %H:%M:%S')
       df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
       ic(df.head())

       # Change all column headings to be lower case, and remove spacing
       df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
       ic(df.head())

       # exit(0)

       # Get month of each sample
       df['month'] = df['date'].dt.month

       # Sort by datetime
       df.sort_values(by='date', inplace=True, ascending=True)

       ic(df.head())

       return(df)

  def diffs(self,df):
      # Get difference between high and low of each day
      df['range_hl'] = df['high'] - df['low']
      df.drop(['high', 'low'], axis=1, inplace=True)

      # Get difference between open and close of each day
      df['range_oc'] = df['open'] - df['close']
      df.drop(['open', 'close'], axis=1, inplace=True)

      ic(df.head())

      return(df)


  # def get_mov_avg_std(df, col, N):
  def get_mov_avg_std(self,df, col):
      """
      Given a dataframe, get mean and std dev at timestep t using values from t-1, t-2, ..., t-N.
      Inputs
           df         : dataframe. Can be of any length.
           col        : name of the column you want to calculate mean and std dev
           N          : get mean and std dev at timestep t using values from t-1, t-2, ..., t-N
      Outputs
           df_out     : same as df but with additional column containing mean and std dev
      """
      mean_list = df[col].rolling(window = self.N, min_periods=1).mean() # len(mean_list) = len(df)
      std_list = df[col].rolling(window = self.N, min_periods=1).std()   # first value will be NaN, because normalized by N-1
    
      # Add one timestep to the predictions
      mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
      std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))
    
      # Append mean_list to df
      df_out = df.copy()
      df_out[col + '_mean'] = mean_list
      df_out[col + '_std'] = std_list
      # ic(df_out)
      # breakpoint()
      return (df_out)

  def get_mape(self,y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


  def splits(self,df):
    # Get sizes of each of the datasets
    num_cv = int(self.cv_size*len(df))
    num_test = int(self.test_size*len(df))
    num_train = len(df) - num_cv - num_test
    print("num_train = " + str(num_train))
    print("num_cv = " + str(num_cv))
    print("num_test = " + str(num_test))

    # Split into train, cv, and test
    train = df[:num_train]
    cv = df[num_train:num_train+num_cv]
    train_cv = df[:num_train+num_cv]
    test = df[num_train+num_cv:]
    print("train.shape = " + str(train.shape))
    print("cv.shape = " + str(cv.shape))
    print("train_cv.shape = " + str(train_cv.shape))
    print("test.shape = " + str(test.shape))

    ic(df)
    return(df,train, cv, train_cv, test)

  def scale(self,df,train, cv, train_cv, test, cols_list):
      cols_to_scale = ["adj_close"]

      for i in range(1,self.N+1):
         cols_to_scale.append("adj_close_lag_"+str(i))
         cols_to_scale.append("range_hl_lag_"+str(i))
         cols_to_scale.append("range_oc_lag_"+str(i))
         cols_to_scale.append("volume_lag_"+str(i))

      # Do scaling for train set
      # Here we only scale the train dataset, and not the entire dataset to prevent information leak
      scaler = StandardScaler()
      train_scaled = scaler.fit_transform(train[cols_to_scale])
      print("scaler.mean_ = " + str(scaler.mean_))
      print("scaler.var_ = " + str(scaler.var_))
      print("train_scaled.shape = " + str(train_scaled.shape))

      # Convert the numpy array back into pandas dataframe
      train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
      train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]
      print("train_scaled.shape = " + str(train_scaled.shape))
      ic(train_scaled.head())

      # Do scaling for train+dev set
      scaler_train_cv = StandardScaler()
      train_cv_scaled = scaler_train_cv.fit_transform(train_cv[cols_to_scale])
      print("scaler_train_cv.mean_ = " + str(scaler_train_cv.mean_))
      print("scaler_train_cv.var_ = " + str(scaler_train_cv.var_))
      print("train_cv_scaled.shape = " + str(train_cv_scaled.shape))

      self.scaler = scaler

      # Convert the numpy array back into pandas dataframe
      train_cv_scaled = pd.DataFrame(train_cv_scaled, columns=cols_to_scale)
      train_cv_scaled[['date', 'month']] = train_cv.reset_index()[['date', 'month']]
      print("train_cv_scaled.shape = " + str(train_cv_scaled.shape))
      # ic(train_cv_scaled.head())

      # Do scaling for dev set
      ic("Do scaling for dev set")
      cv_scaled = cv[['date']]
      ic(cv)
      # for col in tqdm_notebook(cols_list):
      for col in cols_list:
          feat_list = [col + '_lag_' + str(shift) for shift in range(1, self.N+1)]
          temp = cv.apply(lambda row: self.scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
          cv_scaled = pd.concat([cv_scaled, temp], axis=1)

      ic("Now the entire DEV SeT (dev set) is scaled")
      ic(cv_scaled.head())

      # Do scaling for test set
      test_scaled = test[['date']]
      # for col in tqdm_notebook(cols_list):
      for col in cols_list:
           feat_list = [col + '_lag_' + str(shift) for shift in range(1, self.N+1)]
           temp = test.apply(lambda row: self.scale_row(row[feat_list], row[col+'_mean'], row[col+'_std']), axis=1)
           test_scaled = pd.concat([test_scaled, temp], axis=1)

      # Now the entire test set is scaled
      ic("Now the entire TEST SET (test set) is scaled")
      ic(test_scaled.head())


      features = []
      for i in range(1,self.N+1):
          features.append("adj_close_lag_"+str(i))
          features.append("range_hl_lag_"+str(i))
          features.append("range_oc_lag_"+str(i))
          features.append("volume_lag_"+str(i))

      target = "adj_close"


      # Split into X and y
      X_train = train[features]
      y_train = train[target]
      # breakpoint()
      X_cv = cv[features]
      y_cv = cv[target]
      X_train_cv = train_cv[features]
      y_train_cv = train_cv[target]
      X_sample = test[features]
      y_sample = test[target]
      print("X_train.shape = " + str(X_train.shape))
      print("y_train.shape = " + str(y_train.shape))
      print("X_cv.shape = " + str(X_cv.shape))
      print("y_cv.shape = " + str(y_cv.shape))
      print("X_train_cv.shape = " + str(X_train_cv.shape))
      print("y_train_cv.shape = " + str(y_train_cv.shape))
      print("X_sample.shape = " + str(X_sample.shape))
      print("y_sample.shape = " + str(y_sample.shape))

      # Split into X and y
      X_train_scaled = train_scaled[features]
      y_train_scaled = train_scaled[target]
      X_cv_scaled = cv_scaled[features]
      X_train_cv_scaled = train_cv_scaled[features]
      y_train_cv_scaled = train_cv_scaled[target]
      X_sample_scaled = test_scaled[features]
      print("X_train_scaled.shape = " + str(X_train_scaled.shape))
      print("y_train_scaled.shape = " + str(y_train_scaled.shape))
      print("X_cv_scaled.shape = " + str(X_cv_scaled.shape))
      print("X_train_cv_scaled.shape = " + str(X_train_cv_scaled.shape))
      print("y_train_cv_scaled.shape = " + str(y_train_cv_scaled.shape))
      print("X_sample_scaled.shape = " + str(X_sample_scaled.shape))


      # ---- Put in Common Area
      self.X_train_scaled=X_train_scaled
      self.y_train_scaled=y_train_scaled
      self.y_train=y_train
      self.X_cv_scaled=X_cv_scaled
      self.cv=cv
      self.y_cv=y_cv
      self.features=features
      self.X_train_cv_scaled=X_train_cv_scaled
      self.y_train_cv_scaled=y_train_cv_scaled
      self.X_sample_scaled=X_sample_scaled
      self.y_sample=y_sample
      return()

  # def train_regressor(self,X_train_scaled, y_train_scaled,y_train):
  def train_regressor(self,X_train_scaled, y_train_scaled):
      # Create the model
      model = XGBRegressor(seed=self.model_seed,
                     n_estimators=self.n_estimators,
                     max_depth=self.max_depth,
                     learning_rate=self.learning_rate,
                     min_child_weight=self.min_child_weight,
                     subsample=self.subsample,
                     colsample_bytree=self.colsample_bytree,
                     colsample_bylevel=self.colsample_bylevel,
                     gamma=self.gamma)

      # Train the regressor
      model.fit(X_train_scaled, y_train_scaled)

      return(model)

  def pred_on_train(self, model, X_train_scaled, y_train_scaled,y_train):

      # Do prediction on train set
      # --------------------------
      print ("Predictions on train set")
      est_scaled = model.predict(X_train_scaled)
      est = est_scaled * math.sqrt(self.scaler.var_[0]) + self.scaler.mean_[0]

      # breakpoint()
      # Calculate RMSE
      print("RMSE on train set = %0.3f" % math.sqrt(mean_squared_error(y_train, est)))

      # Calculate MAPE
      print("MAPE on train set = %0.3f%%" % self.get_mape(y_train, est))

  def pred_on_test(self, model, X_cv_scaled, cv, y_cv) :
      # breakpoint()
      # Do prediction on test set
      # -------------------------
      est_scaled = model.predict(X_cv_scaled)
      # cv['est_scaled'] = est_scaled
      # df = df.assign(F = s)
      # breakpoint()
      cv = cv.assign(est_scaled = est_scaled )
      # cv['est'] = cv['est_scaled'] * cv['adj_close_std'] + cv['adj_close_mean']
      cv = cv.assign(est = cv['est_scaled'] * cv['adj_close_std'] + cv['adj_close_mean'])

      # Calculate RMSE
      rmse_bef_tuning = math.sqrt(mean_squared_error(y_cv, cv['est']))
      print("RMSE on dev set = %0.3f" % rmse_bef_tuning)

      # Calculate MAPE
      mape_bef_tuning = self.get_mape(y_cv, cv['est'])
      print("MAPE on dev set = %0.3f%%" % mape_bef_tuning)

  def feats (self, train, features, model):
      # View a list of the features and their importance scores
      imp = list(zip(train[features], model.feature_importances_))
      imp.sort(key=lambda tup: tup[1])
      imp[-10:]
      ic(imp[-10:])


  def scale_row(self,row, feat_mean, feat_std):
    """
    Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
    Inputs
        row      : pandas series. Need to scale this.
        feat_mean: mean  
        feat_std : standard deviation
    Outputs
        row_scaled : pandas series with same length as row, but scaled
    """
    # If feat_std = 0 (this happens if adj_close doesn't change over N days), 
    # set it to a small number to avoid division by zero
    feat_std = 0.001 if feat_std == 0 else feat_std
    
    row_scaled = (row-feat_mean) / feat_std
    
    return row_scaled

  def lags(self,df):

      # Add a column 'order_day' to indicate the order of the rows by date
      df['order_day'] = [x for x in list(range(len(df)))]

      # merging_keys
      merging_keys = ['order_day']

      # List of columns that we will use to create lags
      lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume']
      ic(lag_cols)
      ic(df.head())

      shift_range = [x+1 for x in range(self.N)]
      ic(shift_range)
      # exit(0)

      # for shift in tqdm_notebook(shift_range):
      # for shift in tqdm.notebook.tqdm(shift_range):
      for shift in shift_range:
           train_shift = df[merging_keys + lag_cols].copy()
    
           # E.g. order_day of 0 becomes 1, for shift = 1.
           # So when this is merged with order_day of 1 in df, this will represent lag of 1.
           train_shift['order_day'] = train_shift['order_day'] + shift
    
           foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
           train_shift = train_shift.rename(columns=foo)

           df = pd.merge(df, train_shift, on=merging_keys, how='left') #.fillna(0)
    
      del train_shift

      # Remove the first N rows which contain NaNs
      df = df[self.N:]

      ic(df.head())

      cols_list = ["adj_close","range_hl","range_oc","volume"]

      for col in cols_list:
          df = self.get_mov_avg_std(df, col)

      ic(df.head())

      # df.head()

      return(df,cols_list)

  def train_pred_eval_model(self, X_train_scaled, \
                          y_train_scaled, \
                          X_test_scaled, \
                          y_test, \
                          col_mean, \
                          col_std, \
                          seed=100, \
                          n_estimators=100, \
                          max_depth=3, \
                          learning_rate=0.1, \
                          min_child_weight=1, \
                          subsample=1, \
                          colsample_bytree=1, \
                          colsample_bylevel=1, \
                          gamma=0):

          '''
          Train model, do prediction, scale back to original range and do evaluation
          Use XGBoost here.
          Inputs
              X_train_scaled     : features for training. Scaled to have mean 0 and variance 1
              y_train_scaled     : target for training. Scaled to have mean 0 and variance 1
              X_test_scaled      : features for test. Each sample is scaled to mean 0 and variance 1
              y_test             : target for test. Actual values, not scaled.
              col_mean           : means used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
              col_std            : standard deviations used to scale each sample of X_test_scaled. Same length as X_test_scaled and y_test
              seed               : model seed
              n_estimators       : number of boosted trees to fit
              max_depth          : maximum tree depth for base learners
              learning_rate      : boosting learning rate (xgb’s “eta”)
              min_child_weight   : minimum sum of instance weight(hessian) needed in a child
              subsample          : subsample ratio of the training instance
              colsample_bytree   : subsample ratio of columns when constructing each tree
              colsample_bylevel  : subsample ratio of columns for each split, in each level
              gamma              :
          Outputs
              rmse               : root mean square error of y_test and est
              mape               : mean absolute percentage error of y_test and est
              est                : predicted values. Same length as y_test
          '''

          model = XGBRegressor(seed=self.model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)

          # Train the model
          model.fit(X_train_scaled, y_train_scaled)

          # Get predicted labels and scale back to original range
          est_scaled = model.predict(X_test_scaled)
          est = est_scaled * col_std + col_mean

          # Calculate RMSE
          rmse = math.sqrt(mean_squared_error(y_test, est))
          mape = self.get_mape(y_test, est)

          return rmse, mape, est


  def main_Process(self):

      print("Main process of xgboost on VTI")
      df = self.read_data()
      df = self.diffs(df)
      df,cols_list = self.lags(df)
      df,train,cv,train_cv,test = self.splits(df)

      self.scale(df,train,cv,train_cv,test,cols_list)
      X_train_scaled=self.X_train_scaled
      y_train_scaled=self.y_train_scaled
      y_train=self.y_train
      X_cv_scaled=self.X_cv_scaled
      cv=self.cv
      y_cv=self.y_cv
      features=self.features
      X_train_cv_scaled=self.X_train_cv_scaled
      y_train_cv_scaled=self.y_train_cv_scaled
      X_sample_scaled=self.X_sample_scaled
      y_sample=self.y_sample

      # ic(y_train)
      model = self.train_regressor(X_train_scaled, y_train_scaled)
      # ic(y_train)
      self.pred_on_train( model, X_train_scaled, y_train_scaled,y_train)
      self.pred_on_test(model, X_cv_scaled, cv, y_cv)
      self.feats(train , features, model )


      rmse, mape, est = self.train_pred_eval_model(X_train_cv_scaled, 
                             y_train_cv_scaled, 
                             X_sample_scaled, 
                             y_sample, 
                             test['adj_close_mean'],
                             test['adj_close_std'],
                             seed=self.model_seed,
                             # n_estimators=self.n_estimators_opt, 
                             n_estimators=self.n_estimators, 
                             # max_depth=max_depth_opt, 
                             max_depth=self.max_depth, 
                             # learning_rate=learning_rate_opt, 
                             learning_rate=self.learning_rate, 
                             # min_child_weight=min_child_weight_opt, 
                             min_child_weight=self.min_child_weight, 
                             # subsample=subsample_opt, 
                             subsample=self.subsample, 
                             # colsample_bytree=colsample_bytree_opt, 
                             colsample_bytree=self.colsample_bytree, 
                             colsample_bylevel=self.colsample_bylevel, 
                             # colsample_bylevel=colsample_bylevel_opt, 
                             # gamma=gamma_opt)
                             gamma=self.gamma)

      # Calculate RMSE
      print("RMSE on test set = %0.3f" % rmse)

      # Calculate MAPE
      print("MAPE on test set = %0.3f%%" % mape)


# --- Main  ---
if __name__ == '__main__':
    print("Start xgboost on VTI")

    p1 = ct22xgboost("param_data.json")

    p1.main_Process()


