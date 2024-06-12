# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
ts1 = pd.read_csv('TS2.csv',index_col='Day', parse_dates=True, date_parser=dateparse)
# ts1 = pd.read_csv('TS2.csv',index_col='Day', parse_dates=True,delim_whitespace=True)
# ts1 = pd.read_csv('airline-passengers.csv',index_col='Month', parse_dates=True)
# finding shape of the dataframe
print(ts1)
print(ts1.shape)
# having a look at the data
print(ts1.head())
# plotting the original data
# ts1[['Extracted']].plot(title='Passengers Data')
# print(ts1[['Thousands of Passengers']])
print(ts1[['Extracted']])
ts1['Extracted'].plot(title='Extracted')


# decompose_result = seasonal_decompose(ts1['Thousands of Passengers'],model='multiplicative')
decompose_result = seasonal_decompose(ts1['Extracted'],model='multiplicative',period=7)
# decompose_result = seasonal_decompose(ts1['Extracted'],model='multiplicative')
decompose_result.plot();

print ("Decompose result Done")
# exit(0)
# Set the frequency of the date time index as Monthly start as indicated by the data
ts1.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
# m = 12
m = 7
alpha = 1/(2*m)


# Single HWES
ts1['HWES1'] = SimpleExpSmoothing(ts1['Extracted']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
# ts1['HWES1'] = SimpleExpSmoothing(ts1['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
ts1[['Extracted','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');
# ts1[['Thousands of Passengers','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');
