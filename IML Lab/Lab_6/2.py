import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Social_Network_Ads.csv")
print("Missing values in dataset ")
print(df.isnull().sum())

df.columns = ["User ID", "Gender", "Age", "EstimatedSalary", "Purchased"]
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
X = df[[ "Gender", "Age", "EstimatedSalary"]]
y = df['Purchased'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.3 , random_state = 42 
)
k = int(input("Enter value of K : "))
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print("Accuracy of the model is : ", accuracy)