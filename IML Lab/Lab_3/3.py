import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv("newdata.csv")
df.columns = ["Date", "Temp", "Humidity"]
df["Date"] = pd.to_datetime(df["Date"] , errors="coerce")
df = df.dropna()
df["Date"] = (df["Date"] - df["Date"].min()).dt.days
X = df[["Date", "Humidity"]]  
y = df["Temp"]                
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)