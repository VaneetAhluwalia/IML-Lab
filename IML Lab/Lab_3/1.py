import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

da = pd.read_csv("D:/IML lab/data.csv")
print(da.head())

da["Date"] = pd.to_datetime(da["Date"])

da["Date"] = (da["Date"]-da["Date"].min()).dt.days

X = da["Date"]

y = da["Daily minimum temperatures"]


X_train , X_test , y_train , y_test = train_test_split(
    X,y,test_size = 0.2 , random_state = 42
)

model = LinearRegression()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test , y_pred)
print(r2)