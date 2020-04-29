
import csv
from random import seed
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model
import pickle
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.tsatools import lagmat, add_trend
from statsmodels.tsa.adfvalues import mackinnonp
import scipy.stats as stats

def adf(ts, maxlag=1):
    """
    Augmented Dickey-Fuller unit root test
    """
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)
     
    # Get the dimension of the array
    nobs = ts.shape[0]
         
    # Calculate the discrete difference
    tsdiff = np.diff(ts)
     
    # Create a 2d array of lags, trim invalid observations on both sides
    tsdall = lagmat(tsdiff[:, None], maxlag, trim='both', original='in')
    # Get dimension of the array
    nobs = tsdall.shape[0] 
     
    # replace 0 xdiff with level of x
    tsdall[:, 0] = ts[-nobs - 1:-1]  
    tsdshort = tsdiff[-nobs:]
     
    # Calculate the linear regression using an ordinary least squares model    
    results = OLS(tsdshort, add_trend(tsdall[:, :maxlag + 1], 'c')).fit()
    adfstat = results.tvalues[0]
     
    # Get approx p-value from a precomputed table (from stattools)
    pvalue = mackinnonp(adfstat, 'c', N=1)
    return pvalue
 
def cadf(x, y):
    """
    Returns the result of the Cointegrated Augmented Dickey-Fuller Test
    """
    # Calculate the linear regression between the two time series
    ols_result = OLS(x, y).fit()
     
    # Augmented Dickey-Fuller unit root test
    return adf(ols_result.resid)
class Crop():
    def __init__(self, name, latitude, longitude):
        self.name = name        
        self.latitude = latitude
        self.longitude = longitude
        self.price_list = []
        self.date = []
        self.delivery_start_date = []
        self.delivery_end_date = []
    def new_info(self, price, date, delivery_start_date, delivery_end_date):
        self.price_list.append(float(price))
        self.date.append(date)
        self.delivery_start_date.append(datetime.strptime(delivery_start_date, "%m/%d/%Y"))#.date()
        self.delivery_end_date.append(delivery_end_date)
    def printCrop(self):
        print("Crop :" , self.name,self.latitude,  self.longitude) 
    def __eq__(self, other):
        return (self.name == other.name and self.latitude == other.latitude and self.longitude == other.longitude)
seed(1)    
##list_unique_crop = []        
##with open('Query2.csv', newline='') as csvfile:#MerhdadData.csv
##    line = csvfile.readline()
##    for row in range(1,99591):
##        print(row)
##        line = csvfile.readline()
##        lineList = line.split(",")
##        crp = Crop(lineList[1], lineList[3],lineList[4])
##        if crp not in list_unique_crop:
##            crp.new_info(lineList[2],lineList[0],lineList[5],lineList[6])
##            list_unique_crop.append(crp)
##        else :
##            ind = list_unique_crop.index(crp)
##            list_unique_crop[ind].new_info(lineList[2],lineList[0],lineList[5],lineList[6])
##
### Saving the objects:
##with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
##    pickle.dump([list_unique_crop], f)
with open('objs.pkl','rb') as f:  # Python 3: open(..., 'rb')
    list_unique_crop = pickle.load(f)
list_unique_crop = list_unique_crop[0]
farm_ind = 3
p , d , q= 5 , 1, 1
ind_list = list(np.argsort(list_unique_crop[farm_ind].delivery_start_date))
series = np.array(list_unique_crop[farm_ind].price_list, dtype=float)
#list(np.transpose(sorted(list_unique_crop[2].delivery_start_date))),
series = np.transpose(series[ind_list])
all_data = DataFrame()
all_data['price'] = series
all_data['month'] = sorted(list_unique_crop[farm_ind].delivery_start_date)
plt.plot(series)
plt.show()
autocorrelation_plot(series)
plt.show()
### fit model
model = ARIMA(series, order=(p,d,q))
model_fit = model.fit(disp=0)
print(model_fit.summary())
###plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())

size = int(len(series) * 0.66)
train, test = series[0:size], series[size:len(series)]
history = list(train)
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)

print('Test MSE: %.3f' % error)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
AIC = 2*(-np.sum(stats.norm.logpdf(test, predictions)))+ (2*(p+q+d+1))
####### GARCH
train.mean()
squared_data = (train-train.mean())**2
test_var = (test-test.mean())**2
plot_acf(squared_data)
plt.show()
model = arch_model(squared_data, mean='Zero', vol='ARCH', p=15)
model_fit = model.fit()
yhat = model_fit.forecast(horizon=len(test))
plt.plot(test_var)
plt.plot(yhat.variance.values[-1,:])
plt.show()

model = arch_model(squared_data, mean='Zero', vol='GARCH', p=15, q=15)
model_fit = model.fit()
yhat = model_fit.forecast(horizon=len(test))
plt.plot(test_var)
plt.plot(yhat.variance.values[-1,:])
plt.show()
#######Monte Carlo Simulation
train_DataFrame = DataFrame(train)
log_returns = np.log(1+train_DataFrame.pct_change())
train_DataFrame.plot()
plt.show()
log_returns.plot()
plt.show()

u = log_returns.mean()
u
var = log_returns.var()
var
drift = u - (0.5 * var)
drift
stdev = log_returns.std()
stdev
t_intervals = len(test)
iterations = 10
daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
start_price = train[-1]
price_list_simulation = np.zeros_like(daily_returns)
price_list_simulation[0] = start_price
for t in range(1, t_intervals):
    price_list_simulation[t] = price_list_simulation[t - 1] * daily_returns[t]
plt.plot(range(1,len(train)+1),train)
plt.plot(range(len(train)+1, len(train)+len(test)+1),price_list_simulation)
plt.show()
first_part = DataFrame(train)
second_part = DataFrame(price_list_simulation)

#####Cointegration
ind_list2 = list(np.argsort(list_unique_crop[farm_ind].delivery_end_date))
series2 = np.array(list_unique_crop[farm_ind].price_list, dtype=float)
#list(np.transpose(sorted(list_unique_crop[2].delivery_start_date))),
series2 = np.transpose(series2[ind_list2])
all_data2 = DataFrame()
all_data2['price'] = series2
all_data2['month'] = sorted(list_unique_crop[farm_ind].delivery_end_date)
plt.plot(series2)
plt.show()
autocorrelation_plot(series2)
plt.show()
p_value = cadf(series, series2)
p_value
##
### Getting back the objects:
##with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
##    obj0, obj1, obj2 = pickle.load(f)


##plt.plot(list_unique_crop[ind].price_list)
##plt.ylabel('Price')
##plt.xlabel('Date')
##plt.show()

##
##import psycopg2
##try:
##    connection = psycopg2.connect(user = "postgres",
##                                  password = "admin",
##                                  host = "127.0.0.1",
##                                  port = "5432",
##                                  database = "FarmLink")
##
##    cursor = connection.cursor()
##    # Print PostgreSQL Connection properties
##    #print ( connection.get_dsn_parameters(),"\n")
##
##except (Exception, psycopg2.Error) as error :
##    print ("Error while connecting to PostgreSQL", error)
