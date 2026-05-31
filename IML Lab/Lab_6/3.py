import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("titanic.csv")
print("Missing values in dataset ")
print(df.isnull().sum())

df.columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
df = df.drop(["Cabin" , "PassengerId" , "Name" , "Ticket"] , axis = 1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})



X = df[[ "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df['Survived'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size = 0.3 , random_state = 42
)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train , y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test , y_pred)
print("Accuracy of the model is : ", accuracy)