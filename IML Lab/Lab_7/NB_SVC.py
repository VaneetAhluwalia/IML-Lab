import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def run_dataset(file_name):
    print("\nDataset:", file_name)
    df = pd.read_csv(file_name)

    print("Missing values in dataset:")
    print(df.isnull().sum())

    if file_name == "titanic.csv":
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df = df.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'])
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
        X = df[[ "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        y = df['Survived'].values

    elif file_name == "Social_Network_Ads.csv":
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        X = df.drop(['User ID', 'Purchased'], axis=1).values
        y = df['Purchased'].values

    elif file_name == "diabetes_dataset.csv":
        
        df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
        y = df['Outcome'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model1 = GaussianNB()
    model2 = SVC()

    model1.fit(X_train , y_train)
    model2.fit(X_train , y_train)
    y_pred_1 = model1.predict(X_test)
    y_pred_2 = model2.predict(X_test)

    accuracy1 = accuracy_score(y_test , y_pred_1)
    accuracy2 = accuracy_score(y_test , y_pred_2)

    print("Accuracy using the GaussionNB model is : " , accuracy1)
    print("Accuracy using the SVM model is : ", accuracy2)


datasets = ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]

for d in datasets:
    run_dataset(d)