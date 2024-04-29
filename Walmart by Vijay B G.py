#!/usr/bin/env python
# coding: utf-8

# # Capstone Project - Walmart Sales Forcasting
# ## Submited by - Vijay B G - vijay.sudhaganesh@gamil.com

# ## 1. You are provided with the weekly sales data for their various outlets. 

# In[1]:


# Use statistical analysis, EDA, outlier analysis, and handle the missing values to come up with various
# insights that can give them a clear perspective on the following:


# In[2]:


#importing the Libearys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Importing the Walmart Dataset

wal = pd.read_csv("./Walmart.csv")


# In[4]:


wal.head(5)


# In[5]:


#size of the Dataset

wal.shape


# In[6]:


#Total stores given in the Dataset

wal['Store'].unique()


# In[7]:


#Checking the information about the dataset

wal.info()


# In[8]:


#checking the null values present in the dataset

wal.isnull().sum()


# In[9]:


#checking the the statistical data about dataset

wal.describe()


# In[10]:


#seeing the number of store in the dataset

store_numbers=list(wal['Store'].unique())
len(store_numbers)


# In[11]:


# we are seperating the data in stores wise. Different store data is created. 

for i in range(1,len(store_numbers)+1):
    globals()[f'Store{i}'] = wal[wal['Store'] == i]


# In[12]:


#for example, simply type 'Store'end with the store number (1 to 45), it show's the entire data about the store.

Store30


# ### Information about the dataset by ploting the columns.

# In[13]:


#Distribution of the Weekly Sales

plt.figure(figsize =(8,6))
sns.histplot(data = wal, x = 'Weekly_Sales',kde = True,bins =15, color ='b')
plt.xlabel('Median of Weekly scale')
plt.ylabel('Frequency')
plt.title('Distribution of Weekly Sales')
plt.grid(True)
plt.show()


# In[14]:


#Count of the Holiday Flag

plt.figure(figsize =(6,6))
plt.pie(wal['Holiday_Flag'].value_counts(), autopct="%1.1f%%", labels=['Not Holiday','Holiday'])
plt.xlabel('Holiday_Flag')
plt.ylabel('Count')
plt.title('Count of Holiday_Flag')
plt.show()


# In[15]:


#Distribution of the Temperature

plt.figure(figsize =(8,6))
sns.histplot(data = wal, x = 'Temperature',kde = True,bins =15, color ='r')
plt.xlabel('Median of Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Temperature')
plt.grid(True)
plt.show()


# In[16]:


#Distribution of the Fuel_Price

plt.figure(figsize =(8,6))
sns.histplot(data = wal, x = 'Fuel_Price',kde = True,bins =15, color ='g')
plt.xlabel('Median of Fuel_Price')
plt.ylabel('Frequency')
plt.title('Distribution of Fuel_Price')
plt.grid(True)
plt.show()


# In[17]:


#Distribution of the CPI

plt.figure(figsize =(8,6))
sns.histplot(data = wal, x = 'CPI',kde = True,bins =15, color ='y')
plt.xlabel('Median of CPI')
plt.ylabel('Frequency')
plt.title('Distribution of CPI')
plt.grid(True)
plt.show()


# In[18]:


#Distribution of the Unemployment

plt.figure(figsize =(8,6))
sns.histplot(data = wal, x = 'Unemployment',kde = True,bins =15, color ='b')
plt.xlabel('Median of Unemployment')
plt.ylabel('Frequency')
plt.title('Distribution of Unemployment')
plt.grid(True)
plt.show()


# ### a. If the weekly sales are affected by the unemployment rate, if yes - which stores are suffering the most?

# In[19]:


#Let just pick those who have less Weekly Sales. As it will be too lenghty to analyze 45 stores data.
#We will pick those stores for which weekly sales is less than 15 percentile.


# In[20]:


#By using the Quantile to find least weeekly sales data from the dataset.

a1 = round(wal['Weekly_Sales'].quantile(0.15),1)
a1


# In[21]:


less_sales = wal[wal['Weekly_Sales'] < a1]['Store'].unique()
less_sales


# In[22]:


#For the above obtain less sales stores we plot the scatter plot

plt.figure(figsize = (20,15))

plt.subplot(4,3,1)
sns.scatterplot(data = Store3, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,2)
sns.scatterplot(data = Store5, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,3)
sns.scatterplot(data = Store7, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,4)
sns.scatterplot(data = Store16, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,5)
sns.scatterplot(data = Store29, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,6)
sns.scatterplot(data = Store30, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,7)
sns.scatterplot(data = Store33, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,8)
sns.scatterplot(data = Store36, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,9)
sns.scatterplot(data = Store37, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,10)
sns.scatterplot(data = Store38, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,11)
sns.scatterplot(data = Store42, x = "Unemployment", y = "Weekly_Sales" )

plt.subplot(4,3,12)
sns.scatterplot(data = Store44, x = "Unemployment", y = "Weekly_Sales" )


# By seeing the above graph, the two store weelkly sales reduce the unemployement increase those are 38 and 44.
# In some of the store the unemployement increase the weelkly sales also increase those are 5, 30 and 36.
# Remaining graph are seems like average weelkly sales in the period, no unemployement affected.

# ### b. If the weekly sales show a seasonal trend, when and what could be the reason?

# In[23]:


#Converting the Date column dtype to Date-Time.

wal['Date'] = pd.to_datetime(wal['Date'])
wal['Date'].info()


# In[24]:


#Selecting the required column and store into new variable

sales_date = wal[['Date','Weekly_Sales']]
sales_date.set_index('Date',inplace= True)
sales_date


# In[25]:


#ploting the line plot for Period vs Sales.

plt.figure(figsize = (20,7))
sns.lineplot(data = sales_date)

plt.xlabel('Period',fontsize = 20)
plt.ylabel('Weekly Sales', fontsize = 20)
plt.title('Period vs Sales', fontsize = 30)
plt.show()


# In[26]:


#ploting the line plot for Period vs Sales

plt.figure(figsize = (20,7))
sns.lineplot(data = wal, x = 'Date', y ='Holiday_Flag')

plt.xlabel('Period',fontsize = 20)
plt.ylabel('Holiday_flag', fontsize = 20)
plt.title('Period vs Holiday_Flag', fontsize = 30)
plt.show()


# We can clearly see there is a seasonal trend in weekly sales. 
# Whole year sales is average. But at the end of the year there is the holiday season begins
# 
# So, the spike in the sales overlaps with the holiday season.
# 
# AS we known that Walmart famous in weasten country, they celebrate holidays in season of christmans and new year, so the sudden spike in the sales at the end of the year and sales also increase.

# ## c. Does temperature affect the weekly sales in any manner?

# In[27]:


#ploting the line plot for 'Period vs  Temperature and Period vs Sales to compare the Status.

plt.figure(figsize = (20,10))

plt.subplot(2,1,1)
sns.lineplot(data = wal, x = 'Date', y ='Temperature')
plt.xlabel('Period',fontsize = 15)
plt.ylabel('Temperature', fontsize = 15)
plt.title('Period vs  Temperature', fontsize = 20)

plt.figure(figsize = (20,10))
plt.subplot(2,1,2)
sns.lineplot(data = sales_date)
plt.xlabel('Period',fontsize = 15)
plt.ylabel('Weekly Sales', fontsize = 15)
plt.title('Period vs Weekly Sales', fontsize = 20)

plt.show()


# In weasten countries the Holiday season are marked with winters and snow, that increases the needed clothing and stuff.
# Other than this there is no such clear trend of shopping related with temprature.

# ### d. How is the Consumer Price index affecting the weekly sales of various stores?

# In[28]:


#ploting the line plot for Period vs Temperature and Period vs Sales to compare the Status.

plt.figure(figsize = (20,10))
plt.subplot(2,1,1)
sns.lineplot(data = wal, x = 'Date', y ='CPI')
plt.xlabel('Period',fontsize = 15)
plt.ylabel('CPI', fontsize = 15)
plt.title('Period vs  CPI', fontsize = 20)

plt.figure(figsize = (20,10))
plt.subplot(2,1,2)
sns.lineplot(data = sales_date)
plt.xlabel('Period',fontsize = 15)
plt.ylabel('Weekly Sales', fontsize = 15)
plt.title('Period vs Weekly Sales', fontsize = 20)

plt.show()


# When the Consumer Price index increase will not affected the weekly sales, we seeams clearly in the above graph. 

# ### e. Top performing stores according to the historical data.

# In[29]:


#Grouping the data for the average sales of the store.

avg_score_sales = wal.groupby('Store')['Weekly_Sales'].agg('mean')
avg_sales = pd.DataFrame(avg_score_sales)
avg_sales['Weekly_Sales'] = avg_sales['Weekly_Sales']/(avg_sales['Weekly_Sales'].max() - avg_sales['Weekly_Sales'].min())
avg_sales.head()


# In[30]:


#ploting the Bar plot for Store vs Weekly Sales.

plt.figure(figsize = (20,8))
sns.barplot(data = avg_sales, x = avg_sales.index, y ='Weekly_Sales')
plt.xlabel('Stores',fontsize = 15)
plt.ylabel('Weekly Sales', fontsize = 15)
plt.title('Store vs Weekly Sales', fontsize = 20)
plt.show()


# In[31]:


#Top 5 Weekly sales stores

avg_sales.sort_values('Weekly_Sales',ascending = False ).head(5)


# By using the given historical data we see the 20,4,14,13,2 thses store are performing well.

# ### f. The worst performing store, and how significant is the difference between the highest and lowest performing stores.

# In[32]:


#Top 5 worse Performing Store.

avg_sales.sort_values('Weekly_Sales',ascending = True ).head(5)


# In[33]:


#ploting the line plot for total period vs Weekly Sales.

plt.figure(figsize = (20,10))
sns.lineplot(data = Store20, x= 'Date', y ='Weekly_Sales', color = '#31fc03')
sns.lineplot(data = Store33, x= 'Date', y ='Weekly_Sales', color = '#fc0377')
plt.xlabel('Date',fontsize = 15)
plt.ylabel('Weekly Sales', fontsize = 15)
plt.title('Date vs Weekly Sales', fontsize = 20)
plt.show()


# In[34]:


#The difference between the highest and lowest performing stores.

diff_higf_to_low=pd.DataFrame(avg_score_sales)


# In[35]:


(diff_higf_to_low.loc[33][0]/diff_higf_to_low.loc[20][0])*100


# Lowest performing store's sales only accounts for 12.3% of sales done by top performing store on average.

# In[36]:


#The least performing store is store 33, we finding the over all percentage of performing store.

sum_of_all_sales = avg_score_sales.sum()


# In[37]:


sum_of_sales_for_33th_shop = avg_score_sales.loc[33]


# In[38]:


per_of_total_sales_to_33_shop = (sum_of_sales_for_33th_shop/sum_of_all_sales)*100


# In[39]:


per_of_total_sales_to_33_shop


# Store no.33 is 0.5 percenrtage of weelky sales.

# ### In the hole Walmart store No.33 is performing less, it about 0.55 percentage only.

# In[ ]:





# ## 2. Use predictive modeling techniques to forecast the sales for each store for the next 12 weeks

# In[40]:


#importing the Libearys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[41]:


data = pd.read_csv('./Walmart.csv')


# In[42]:


data


# In[43]:


#seeing the number of store in the dataset

store_numbers=list(data['Store'].unique())
len(store_numbers)


# In[44]:


#Converting the data to store wise

for i in range(1,len(store_numbers)+1):
    globals()[f'sales{i}'] = data[data['Store'] == i][['Date','Weekly_Sales']]


# In[45]:


#Importing the required libeary

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[46]:


len(store_numbers)


# ### Solving all the 45 store are more time consuming process, so we took minimum 5 stores from the data and do the forecast

# #### Selected store are 1, 14, 20, 33,44 (4,20 - top performing store and 33,44 - worst performing store)

# In[47]:


#Selecting the Store 1

sales1


# In[48]:


#converting the Date column as index

sales1.index = sales1['Date']
del sales1['Date']
sales1


# In[49]:


#ploting the lineplot for Store 1 weekly sales

sales1.plot()


# In[50]:


# Rolling Mean Method and Rolling std. dev method for the Store 1.

mean_log=sales1.rolling(window=12).mean() 
std_log=sales1.rolling(window=12).std()  


# In[51]:


#plotting the line plot for the Rolling Mean and Rolling std. dev

plt.plot(sales1,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[52]:


# checking the stationarity of the time series data and adfuller - Augmented Dickey Fuller

result=adfuller(sales1['Weekly_Sales'])
print(result)


# In[53]:


# The above value is greater than 0.05, so the Data is non stationary.
# Log-Transformation Method of converting the non stationary data into stationary data

sales1_log=np.log(sales1)
sales1_log=sales1_log.dropna()
sales1_log.plot()


# In[54]:


# Again plotting the line plot for the Rolling Mean and Rolling std. dev to see the difference of stationary data

mean_log=sales1_log.rolling(window=12).mean() 
std_log=sales1_log.rolling(window=12).std()  

plt.plot(sales1_log,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[55]:


#Again adfuller for stationary data

result2=adfuller(sales1_log['Weekly_Sales'])
print(result2)


# In[56]:


# first_log is the variable giving the plot for the original time series data after normalizaing it using log method
# mean_log to check the stationarity
# first_log and mean_log difference

sales1_new=sales1_log-mean_log
sales1_new=sales1_new.dropna()
sales1_new.plot()


# In[57]:


# After the process the shape of the data

sales1_new.shape


# In[58]:


# plotting the decompose_result

decompose_result=seasonal_decompose(sales1_new['Weekly_Sales'],period=12)
decompose_result.plot()


# In[59]:


#Plotting the acf and pacf plot

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(sales1_new['Weekly_Sales'], ax=ax[0],)
plot_pacf(sales1_new['Weekly_Sales'], ax=ax[1],)
plt.show()


# In[60]:


#model splitting the train and test data

train=sales1_new.iloc[:106]['Weekly_Sales']
test=sales1_new.iloc[107:]['Weekly_Sales']


# In[61]:


#ARIMA MODEL
#the value of p=1,d=0,q=5 we took the data from the above acf and pacf plot.

model = ARIMA(train, order=(1, 0, 5)) #p=1,d=0,q=5 from the acf and pacf plot
model_fit = model.fit()


# In[62]:


model_fit


# In[63]:


#See the forcast of ARIMA MODEL

forecast = model_fit.forecast(steps=26) 
sales1_new.plot()
forecast.plot()


# ### In ARIMA model the forecast not performing well so did SARIMAX MODEL

# In[64]:


#SARIMAX MODEL

model1 = SARIMAX(train, order=(1, 0, 5),seasonal_order=(1,0,5,26))  #p=1,d=0,q=5 from the acf and pacf plot
model1_fit = model1.fit()


# In[65]:


model1_fit


# In[66]:


#See the forcast of SARIMAX MODEL

forecast = model1_fit.forecast(steps=26)  
sales1_new.plot()
forecast.plot()


# ### Forecast the sales for Store 1 for the next 12 weeks.

# In[67]:


# Forcast of next 12 weeks data of store 1

forecast = model1_fit.forecast(steps=26+12)  # Forecasting for the next 12 periods
sales1_new.plot()
forecast.plot()


# ## Store 14

# In[68]:


#Selecting the Store 14

sales14


# In[69]:


#converting the Date column as index

sales14.index = sales14['Date']
del sales14['Date']
sales14


# In[70]:


#ploting the lineplot for Store 14 weekly sales

sales14.plot()


# In[71]:


# Rolling Mean Method and Rolling std. dev method for the Store 14.

mean_log=sales14.rolling(window=12).mean() 
std_log=sales14.rolling(window=12).std()  


# In[72]:


#plotting the line plot for the Rolling Mean and Rolling std. dev

plt.plot(sales14,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[73]:


# checking the stationarity of the time series data and adfuller - Augmented Dickey Fuller

result=adfuller(sales14['Weekly_Sales'])
print(result)


# In[74]:


# The above value is greater than 0.05, so the Data is non stationary.
# Log-Transformation Method of converting the non stationary data into stationary data

sales14_log=np.log(sales14)
sales14_log=sales14_log.dropna()
sales14_log.plot()


# In[75]:


# Again plotting the line plot for the Rolling Mean and Rolling std. dev to see the difference of stationary data

mean_log=sales14_log.rolling(window=12).mean() 
std_log=sales14_log.rolling(window=12).std()  

plt.plot(sales14_log,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[76]:


#Again adfuller for stationary data

result2=adfuller(sales14_log['Weekly_Sales'])
print(result2)


# In[77]:


# first_log is the variable giving the plot for the original time series data after normalizaing it using log method
# mean_log to check the stationarity
# first_log and mean_log difference

sales14_new=sales14_log-mean_log
sales14_new=sales14_new.dropna()
sales14_new.plot()


# In[78]:


# After the process the shape of the data

sales14_new.shape


# In[79]:


# plotting the decompose_result

decompose_result=seasonal_decompose(sales14_new['Weekly_Sales'],period=12)
decompose_result.plot()


# In[80]:


#Plotting the acf and pacf plot

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(sales14_new['Weekly_Sales'], ax=ax[0],)
plot_pacf(sales14_new['Weekly_Sales'], ax=ax[1],)
plt.show()


# In[81]:


#model splitting the train and test data

train=sales14_new.iloc[:106]['Weekly_Sales']
test=sales14_new.iloc[107:]['Weekly_Sales']


# In[82]:


#ARIMA MODEL
#the value of p=1,d=0,q=3 we took the data from the above acf and pacf plot.

model = ARIMA(train, order=(1, 0, 3))  # p=1,d=0,q=3 from the above acf and pacf plot
model_fit = model.fit()


# In[83]:


model_fit


# In[84]:


#See the forcast of ARIMA MODEL

forecast = model_fit.forecast(steps=26)
sales14_new.plot()
forecast.plot()


# ### In ARIMA model the forecast not performing well so did SARIMAX MODEL

# In[85]:


#SARIMAX MODEL

model1 = SARIMAX(train, order=(1, 0, 3),seasonal_order=(1,0,3,26))  # p=1,d=0,q=3 from the above acf and pacf plot
model1_fit = model1.fit()


# In[86]:


model1_fit


# In[87]:


#See the forcast of SARIMAX MODEL

forecast = model1_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=True)  
sales14_new.plot()
forecast.plot()


# ### Forecast of weekly sale for next 12 weeks for store 14 

# In[88]:


# Forcast of next 12 weeks data of store 14

forecast = model1_fit.forecast(steps=26+12)  # Forecasting for the next 12 periods
sales14_new.plot()
forecast.plot()


# In[ ]:





# ## Store 20

# In[89]:


#Selecting the Store 20

sales20


# In[90]:


#converting the Date column as index

sales20.index = sales20['Date']
del sales20['Date']
sales20


# In[91]:


#ploting the lineplot for Store 20 weekly sales

sales20.plot()


# In[92]:


# Rolling Mean Method and Rolling std. dev method for the Store 20.

mean_log=sales20.rolling(window=12).mean() 
std_log=sales20.rolling(window=12).std()  


# In[93]:


#plotting the line plot for the Rolling Mean and Rolling std. dev

plt.plot(sales20,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[94]:


# checking the stationarity of the time series data and adfuller - Augmented Dickey Fuller

result=adfuller(sales20['Weekly_Sales'])
print(result)


# In[95]:


# The above value is greater than 0.05, so the Data is non stationary.
# Log-Transformation Method of converting the non stationary data into stationary data

sales20_log=np.log(sales20)
sales20_log=sales20_log.dropna()
sales20_log.plot()


# In[96]:


# Again plotting the line plot for the Rolling Mean and Rolling std. dev to see the difference of stationary data

mean_log=sales20_log.rolling(window=12).mean() 
std_log=sales20_log.rolling(window=12).std()  

plt.plot(sales20_log,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[97]:


#Again adfuller for stationary data

result2=adfuller(sales20_log['Weekly_Sales'])
print(result2)


# In[98]:


# first_log is the variable giving the plot for the original time series data after normalizaing it using log method
# mean_log to check the stationarity
# first_log and mean_log difference

sales20_new=sales20_log-mean_log
sales20_new=sales20_new.dropna()
sales20_new.plot()


# In[99]:


# After the process the shape of the data

sales20_new.shape


# In[100]:


# plotting the decompose_result

decompose_result=seasonal_decompose(sales20_new['Weekly_Sales'],period=12)
decompose_result.plot()


# In[101]:


#Plotting the acf and pacf plot

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(sales20_new['Weekly_Sales'], ax=ax[0],)
plot_pacf(sales20_new['Weekly_Sales'], ax=ax[1],)
plt.show()


# In[102]:


#model splitting the train and test data

train=sales20_new.iloc[:106]['Weekly_Sales']
test=sales20_new.iloc[107:]['Weekly_Sales']


# In[103]:


#ARIMA MODEL
#the value of p=1,d=0,q=3 we took the data from the above acf and pacf plot.

model = ARIMA(train, order=(1, 0, 3))  # p=1,d=0,q=3 from the above acf and pacf plot
model_fit = model.fit()


# In[104]:


model_fit


# In[105]:


#See the forcast of ARIMA MODEL

forecast = model_fit.forecast(steps=26)
sales20_new.plot()
forecast.plot()


# ### In ARIMA model the forecast not performing well so did SARIMAX MODEL

# In[106]:


#SARIMAX MODEL

model1 = SARIMAX(train, order=(1, 0, 3),seasonal_order=(1,0,3,26))  #p=1,d=0,q=3 from the above acf and pacf plot
model1_fit = model1.fit()


# In[107]:


model1_fit


# In[108]:


#See the forcast of SARIMAX MODEL

forecast = model1_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=True)  
sales20_new.plot()
forecast.plot()


# ### Forecast of weekly sale for next 12 weeks for store 20

# In[109]:


# Forcast of next 12 weeks data of store 20

forecast = model1_fit.forecast(steps=26+12)  # Forecasting for the next 12 periods
sales20_new.plot()
forecast.plot()


# In[ ]:





# ## Store 33

# In[110]:


#Selecting the Store 33

sales33


# In[111]:


#converting the Date column as index

sales33.index = sales33['Date']
del sales33['Date']
sales33


# In[112]:


#ploting the lineplot for Store 33 weekly sales

sales33.plot()


# In[113]:


# Rolling Mean Method and Rolling std. dev method for the Store 33.

mean_log=sales33.rolling(window=12).mean() 
std_log=sales33.rolling(window=12).std()  


# In[114]:


#plotting the line plot for the Rolling Mean and Rolling std. dev

plt.plot(sales33,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[115]:


# checking the stationarity of the time series data and adfuller - Augmented Dickey Fuller

result=adfuller(sales33['Weekly_Sales'])
print(result)


# In[116]:


#The value is lesser than 0.05 so the data is stationery.
# After the process the shape of the data

sales33.shape


# In[117]:


# plotting the decompose_result

decompose_result=seasonal_decompose(sales33['Weekly_Sales'],period=12)
decompose_result.plot()


# In[118]:


#Plotting the acf and pacf plot

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(sales33['Weekly_Sales'], ax=ax[0],)
plot_pacf(sales33['Weekly_Sales'], ax=ax[1],)
plt.show()


# In[119]:


#model splitting the train and test data

train=sales33.iloc[:114]['Weekly_Sales']
test=sales33.iloc[115:]['Weekly_Sales']


# In[120]:


#ARIMA MODEL
#the value of p=1,d=0,q=2 we took the data from the above acf and pacf plot.

model = ARIMA(train, order=(1, 0, 2)) #p=1,d=0,q=2 from the above acf and pacf plot
model_fit = model.fit()


# In[121]:


model_fit


# In[122]:


#See the forcast of ARIMA MODEL

forecast = model_fit.forecast(steps=26)
sales33.plot()
forecast.plot()


# ### In ARIMA model the forecast not performing well so did SARIMAX MODEL

# In[123]:


#SARIMAX MODEL

model1 = SARIMAX(train, order=(1, 0, 2),seasonal_order=(1,0,2,26))  ##p=1,d=0,q=2 from the above acf and pacf plot
model1_fit = model1.fit()


# In[124]:


model1_fit


# In[125]:


#See the forcast of SARIMAX MODEL

forecast = model1_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=True)  
sales33.plot()
forecast.plot()


# ### Forecast of weekly sale for next 12 weeks for store 33 

# In[126]:


# Forcast of next 12 weeks data of store 33

forecast = model1_fit.forecast(steps=26+12)  # Forecasting for the next 12 periods
sales33.plot()
forecast.plot()


# In[ ]:





# # Shop 44

# In[127]:


#Selecting the Store 44

sales44


# In[128]:


#converting the Date column as index

sales44.index = sales44['Date']
del sales44['Date']
sales44


# In[129]:


#ploting the lineplot for Store 44 weekly sales

sales44.plot()


# In[130]:


# Rolling Mean Method and Rolling std. dev method for the Store 44.

mean_log=sales44.rolling(window=12).mean() 
std_log=sales44.rolling(window=12).std()  


# In[131]:


#plotting the line plot for the Rolling Mean and Rolling std. dev

plt.plot(sales44,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[132]:


# checking the stationarity of the time series data and adfuller - Augmented Dickey Fuller

result=adfuller(sales44['Weekly_Sales'])
print(result)


# In[133]:


# The above value is greater than 0.05, so the Data is non stationary.
# Log-Transformation Method of converting the non stationary data into stationary data

sales44_log=np.log(sales44)
sales44_log=sales44_log.dropna()
sales44_log.plot()


# In[134]:


# Again plotting the line plot for the Rolling Mean and Rolling std. dev to see the difference of stationary data

mean_log=sales44_log.rolling(window=12).mean() 
std_log=sales44_log.rolling(window=12).std()  

plt.plot(sales44_log,color='blue',label='Original Time Series Data')
plt.plot(mean_log,color='red',label='Rolling Mean Time Series Data')
plt.plot(std_log,color='green',label='Rolling Std. Dev Time Series Data')
plt.title('Comparison of Time Series Data')


# In[135]:


#Again adfuller for stationary data

result2=adfuller(sales44_log['Weekly_Sales'])
print(result2)


# In[136]:


# first_log is the variable giving the plot for the original time series data after normalizaing it using log method
# mean_log to check the stationarity
# first_log and mean_log difference

sales44_new=sales44_log-mean_log
sales44_new=sales44_new.dropna()
sales44_new.plot()


# In[137]:


# After the process the shape of the data

sales44_new.shape


# In[138]:


# plotting the decompose_result

decompose_result=seasonal_decompose(sales44_new['Weekly_Sales'],period=12)
decompose_result.plot()


# In[139]:


#Plotting the acf and pacf plot

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(sales44_new['Weekly_Sales'], ax=ax[0],)
plot_pacf(sales44_new['Weekly_Sales'], ax=ax[1],)
plt.show()


# In[140]:


#model splitting the train and test data

train=sales44_new.iloc[:106]['Weekly_Sales']
test=sales44_new.iloc[107:]['Weekly_Sales']


# In[141]:


#ARIMA MODEL
#the value of p=2,d=0,q=1 we took the data from the above acf and pacf plot.

model = ARIMA(train, order=(2, 0, 1))  #p=2,d=0,q=1 from the above acf and pacf plot
model_fit = model.fit()


# In[142]:


model_fit


# In[143]:


#See the forcast of ARIMA MODEL

forecast = model_fit.forecast(steps=26)
sales44_new.plot()
forecast.plot()


# ### In ARIMA model the forecast not performing well so did SARIMAX MODEL

# In[144]:


#SARIMAX MODEL

model1 = SARIMAX(train, order=(2, 0, 1),seasonal_order=(2,0,1,26))  #p=2,d=0,q=1 from the above acf and pacf plot
model1_fit = model1.fit()


# In[145]:


model1_fit


# In[146]:


#See the forcast of SARIMAX MODEL

forecast = model1_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=True)  
sales44_new.plot()
forecast.plot()


# ### Forecast of weekly sale for next 12 weeks for store 44 

# In[147]:


# Forcast of next 12 weeks data of store 44

forecast = model1_fit.forecast(steps=26+12)  # Forecasting for the next 12 periods
sales44_new.plot()
forecast.plot()


# In[ ]:




