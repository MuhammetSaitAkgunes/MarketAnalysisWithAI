# PLEASE USE JUPYTER NOTEBOOK AND RUN STEP BY STEP.

import pandas as pd
import seaborn as sbn
import numpy as np

dataFrame = pd.read_excel("C:/Users/Muhammet Sait/Desktop/2_yillik.xlsx")

# -----

from sklearn.model_selection import train_test_split

satis_fiyat = dataFrame["satis"].values
degerler = dataFrame[["dis_borc","fiyat_endeks","ihracat","ithalat","faiz_1","faiz_2","faiz_3","faiz_4","faiz_5","faiz_6"]].values

fiyat_train, fiyat_test, deger_train, deger_test = train_test_split(satis_fiyat,degerler,test_size=0.2,random_state=15)

# -----

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(degerler)
deger_train = scaler.transform(deger_train)
deger_test = scaler.transform(deger_test)

# -----

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(27,activation="relu"))
model.add(Dense(29,activation="relu"))
model.add(Dense(15,activation="relu"))
model.add(Dense(17,activation="relu"))
model.add(Dense(21,activation="relu"))

model.add(Dense(1))

model.compile(optimizer="rmsprop",loss="mse")

# -----

model.fit(deger_train, fiyat_train, epochs=500)

# -----

loss = model.history.history["loss"]
sbn.lineplot(x=range(len(loss)),y = loss)

# -----

trainLoss = model.evaluate(deger_train,fiyat_train, verbose=0)
testLoss = model.evaluate(deger_test,fiyat_test, verbose=0)

# -----

trainLoss

# -----

testLoss

# -----

testTahminleri = model.predict(deger_test)

tahminDf = pd.DataFrame(fiyat_test,columns=["Gerçek Fiyat"])
testTahminleri = pd.Series(testTahminleri.reshape(147,))
tahminDf = pd.concat([tahminDf,testTahminleri],axis=1)
tahminDf.columns=["Gerçek Fiyat","Tahmin Fiyat"]

# reshape kısmındaki 147 değişebilir. girdiğin veri sayısına göre güncellemen gerek.

# -----

sbn.scatterplot(x = "Gerçek Fiyat", y = "Tahmin Fiyat", data = tahminDf)

# -----

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(tahminDf["Gerçek Fiyat"], tahminDf["Tahmin Fiyat"])