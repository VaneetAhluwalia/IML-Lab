import pandas as pd
import numpy as np

class LR:
    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]   # add intercept
        self.coef = np.linalg.pinv(X) @ y

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X @ self.coef

    def r2(self, y, y_pred):
        return 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
df = pd.read_csv("newdata.csv")
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna()
df["t"] = (df["Date"] - df["Date"].min()).dt.days
X = df[["t", "Humidity"]].values
y = df["Daily minimum temperatures"].values
model = LR()
model.fit(X, y)
y_pred = model.predict(X)
print("R2 Score:", model.r2(y, y_pred))