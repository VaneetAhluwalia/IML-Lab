import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes_dataset.csv")
print("Missing values in dataset ")
print(df.isnull().sum())

df.columns = ["Pregnancie", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
              "DiabetesPedigreeFunction", "Age", "Outcome"]

X = df[["Pregnancie" , "Glucose" , "BloodPressure" , "SkinThickness" , "Insulin" , "BMI" , 
        "DiabetesPedigreeFunction" , "Age"]]
y = df['Outcome']

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.3 , random_state = 42
)
scaler = StandardScaler()
X = scaler.fit_transform(X)

k = int(input("Enter value of K : "))
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print("Accuracy of the model is : ", accuracy)