import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

def run_dataset(file_name):
    print("\nDataset : " , file_name)
    df = pd.read_csv(file_name)
    
    if file_name == "titanic.csv":
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df = df.drop(columns=['Cabin', 'Sex', 'Name', 'Ticket' , 'Embarked','PassengerId'])
        X = df[[ "Pclass", "Age", "SibSp", "Parch", "Fare"]]
        y = df['Survived'].values

    elif file_name == "Social_Network_Ads.csv":
        df = df.drop(columns=['Gender'])
        X = df.drop(['User ID', 'Purchased'], axis=1)
        y = df['Purchased'].values

    elif file_name == "diabetes_dataset.csv":
        df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
        y = df['Outcome'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X)

    accuracy = accuracy_score(y, model.labels_)
    print("Accuracy: ", accuracy)
        

for d in ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]:
    run_dataset(d)