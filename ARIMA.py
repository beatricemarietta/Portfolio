import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/MyDrive/Data Permintaan Darah AB.xlsx')
df.head()

#Data Visualization
from pylab import rcParams
rcParams['figure.figsize'] = 16, 9
df.plot()

from pylab import rcParams
rcParams['figure.figsize'] = 16, 9
df.plot()
#BOX-COX Transformation (Data Stationarity in Variance)
def box_cox_test(dataset):
  fitted_data, lambda_value = stats.boxcox(dataset)
  if round(lambda_value,1) == 1:
    print('Lambda Value:', round(lambda_value,1), '(Data is stationary in variance)')
  else:
    print('Lambda Value:', round(lambda_value,1), '(Data is not stationary in variance)')
    print('\nFitted Data: \n', fitted_data)
  return fitted_data, lambda_value

fitt_data_t1, fitt_lambda_t1 = box_cox_test(df['AB'])
fitt_data_t2, fitt_lambda_t2 = box_cox_test(fitt_data_t1)

from statsmodels.tsa.stattools import adfuller
#Augmented Dickey-Fuller (ADF) Test (Data Stationarity in Mean)
def adf_test(dataset):
  adftest = adfuller(dataset, maxlag = 0)
  print("1. ADF : ",adftest[0])
  print("2. p-value : ", adftest[1])
  print("3. Num Of Lags : ", adftest[2])
  print("4. Num Of Observations Used For ADF Regression:", adftest[3])
  print("5. Critical Values :")
  for key, val in adftest[4].items():
    print("\t",key, ": ", val)

  print('\nHypothesis Test:')
  if abs(adftest[0]) >= adftest[4]['5%'] or adftest[1] <= 0.05:
    print('Reject null hypothesis, Data is stationary in mean')
  else:
    print('Do not reject null hypothesis, data is not stationary in mean')

adf_test(fitt_data_t1)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Plot ACF
fig, ax1 = plt.subplots(figsize=(12, 4))
plot_acf(fitt_data_t1, lags=20, ax=ax1)
plt.title('ACF Plot')
plt.show()

# Plot PACF
fig, ax2 = plt.subplots(figsize=(12, 4))
plot_pacf(fitt_data_t1, lags=20, ax=ax2)
plt.title('PACF Plot')
plt.show()

pip install pmdarima
import pmdarima as pm
from pmdarima import auto_arima
arima_model = pm.auto_arima(fitt_data_t1,
                      start_p=0, start_q=0,
                      max_p=1, max_q=1,
                      m=0,
                      d=0,
                      seasonal=False,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(arima_model.summary())

import statsmodels.api as sm
model101 = sm.tsa.ARIMA(fitt_data_t1, order=(1,0,1))
modelfit101 = model101.fit()

print(modelfit101.summary())

#-------------------------------------------------------------------------------#
#Ljung Box
#H0 : Tidak ada autokorelasi pd residual
#Tolak H0 jika p-val < sig
#Kesimpulan H0 diterima
#-------------------------------------------------------------------------------#
#Jarque Bera
#H0: Data berdistribusi normal
#Tolak H0 jika p-val < sig
#Kesimpulan H0 diterima
#-------------------------------------------------------------------------------#
#Non Heteroskedastisitas
#H0: tidak ada hetero, varians konstan
#Tolak H0 jika p-val < sig
#Kesimpulan H0 diterima
#-------------------------------------------------------------------------------#

modelfit101.plot_diagnostics(figsize=(10,8))
plt.show()

from scipy.special import inv_boxcox
#Prediction
pred = modelfit101.forecast(12)
pred_actual = inv_boxcox(pred, 0.2)
np.round(pred_actual,2)

data_test = np.array([13,22,28,19,22,19,25,25,12,3,9,26])

#Prediction Accuracy
import statistics
mape = statistics.mean(abs((data_test - predict_actual)/data_test))*100
print("MAPE:", mape,"%")

#Data Visualization Actual VS Prediction
data = np.concatenate([df['AB'],data_test])
pred_index = np.arange(231, 243)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(data)), data, linewidth=0.8, color='#61677A', label="Data Aktual")
plt.plot(pred_index, pred_actual, linewidth=2, color='#D71313', label="Data Prediksi")
plt.legend()
plt.title('Data Aktual VS Data Prediksi\nPermintaan Darah Golongan AB')
plt.show()
