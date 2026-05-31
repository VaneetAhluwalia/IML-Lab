import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class LR:
    def fit(self, X, y):
        X = np.array(X).flatten()
        y = np.array(y)
        xm, ym = X.mean(), y.mean()
        self.m = np.sum((X - xm)*(y - ym)) / np.sum((X - xm)**2)
        self.b = ym - self.m * xm

    def predict(self, X):
        return self.m * np.array(X).flatten() + self.b

    def r2(self, y, y_hat):
        return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna()
df["d"] = (df["Date"] - df["Date"].min()).dt.days
X = df[["d"]]
y = df["Daily minimum temperatures"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=42)
model = LR()
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print("R2 Score:", model.r2(yte, pred))