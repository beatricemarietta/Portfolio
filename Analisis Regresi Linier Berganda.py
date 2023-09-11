import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
import math

df=pd.read_excel('/content/drive/MyDrive/PORTFOLIO/DATA PORTFOLIO/Data Kemiskinan.xlsx')
df.head()

df = df.drop(columns=['Wilayah Jateng'])
df.info()

#Model Regresi Linier Berganda
X=df[['TPT','AMH','Inflasi']]
Y=df['PPM']
X_c=sm.add_constant(X)
anareg_model=sm.OLS(Y,X_c).fit()
print(anareg_model.summary())

#Uji Asumsi
#Uji Linieritas - RAMSEY RESET TEST
Y_pred = anareg_model.predict(X_c)
X_additional = X_c.copy()
X_additional['Y_pred_squared'] = Y_pred**2
X_additional['Y_pred_cubed'] = Y_pred**3
ramsey_model = sm.OLS(Y,X_additional).fit()
df1 = (len(X_additional.columns)-len(X_c.columns))
df2 = (len(df)-len(X_additional.columns))
ramsey_statistic = ((ramsey_model.rsquared-anareg_model.rsquared)/df1)/((1-ramsey_model.rsquared)/df2)
p_value_ramsey = 1 - stats.f.cdf(ramsey_statistic, df1, df2)
print('====== Ramsey RESET Test ======')
print('F statistics       :', round(ramsey_statistic,3))
print('p-value            :', round(p_value_ramsey,3))
print('degree of freedom  :', '(',df1,',',df2,')')
print('===============================')
#Korelasi antar variabel bebas
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

#Uji Normalitas Residual - JARQUE-BERA TEST
residuals = anareg_model.resid
jb_statistic, p_value_jbstat = stats.jarque_bera(residuals)
print('===== Jarque-Bera Test =====')
print('Jarque-Bera          :', round(jb_statistic,3))
print('p-value              :', round(p_value_jbstat,3))
print('============================')

#Uji Non Multikolinieritas - VARIANCE INFLATION FACTOR
vif = pd.DataFrame()
vif["Variable"] = X_c.columns
vif["VIF"] = [variance_inflation_factor(X_c.values, i) for i in range(X_c.shape[1])]
print('========= VIF =========')
print(vif)
print('=======================')

#Uji Non Heteroskedastisitas - WHITE TEST
white = het_white(anareg_model.resid, anareg_model.model.exog)
white_statistic = white[0]
p_value_white = white[1]
print('===== White Test =====')
print('White         :', round(white_statistic,3))
print('p-value       :', round(p_value_white,3))
print('======================')

#Uji Non Autokorelasi - DURBIN-WATSON TEST
durbin_watson_statistic = sm.stats.stattools.durbin_watson(anareg_model.resid)
print('===== Durbin-Watson Test =====')
print('Durbin-Watson          :', round(durbin_watson_statistic,3))
print('==============================')
