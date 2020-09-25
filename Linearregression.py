import quandl
import pandas as pd
import numpy as np
from sklearn import preprocessing ,svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
df = quandl.get("WIKI/GOOGL")
df=df[["Adj. Open","Adj. High","Adj. Low","Adj. Close","Adj. Volume"]]
#               Open    High     Low  ...   Adj. Low  Adj. Close  Adj. Volume
# Date                                ...
# 2004-08-19  100.01  104.06   95.96  ...  48.128568   50.322842   44659000.0
# 2004-08-20  101.01  109.08  100.50  ...  50.405597   54.322689   22834300.0
# 2004-08-23  110.76  113.48  109.05  ...  54.693835   54.869377   18256100.0
# 2004-08-24  111.24  111.60  103.57  ...  51.945350   52.597363   15247300.0
# 2004-08-25  104.76  108.00  103.88  ...  52.100830   53.164113    9188600.0
#
# [5 rows x 12 columns]

df["HL_PCT"]=((df["Adj. High"]-df["Adj. Close"])/df["Adj. Close"])*100  #High minus low percentage
df["PCT_change"]=((df["Adj. Close"]-df["Adj. Open"])/df["Adj. Open"])*100

df=df[["Adj. Close","HL_PCT","PCT_change","Adj. Volume"]]
print(df.head())

#             Adj. Close    HL_PCT  PCT_change  Adj. Volume
# Date
# 2004-08-19   50.322842  3.712563    0.324968   44659000.0
# 2004-08-20   54.322689  0.710922    7.227007   22834300.0
# 2004-08-23   54.869377  3.729433   -1.227880   18256100.0
# 2004-08-24   52.597363  6.417469   -5.726357   15247300.0
# 2004-08-25   53.164113  1.886792    1.183658    9188600.0

forecast_col="Adj. Close"
df.fillna(-99999,inplace=True)  #Getting rid of Nan data if there is

forecast_out=int(math.ceil(0.01*len(df)))     #Just rounds up decimal numbers , for ex: 342432.2 ---> 342432

df["label"]=df[forecast_col].shift(-forecast_out) # Shifting for 10 days back (not sure ,search)
df.dropna(inplace=True) # used to remove rows and columns with Null/NaN values.

X=np.array(df.drop(["label"]),1,inplace=True)
y=np.array(df["label"])

X=preprocessing.scale(X) # We use preprocessing to SCALE values compared with each others so nothing will be neglected
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
cf=LinearRegression()
cf.fit(X_train,y_train)
accuracy=cf.score(X_test,y_test)
print(accuracy)
