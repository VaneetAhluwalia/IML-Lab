import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def run_dataset(file_name):
    print("\n Dataset : " , file_name)
    df = pd.read_csv(file_name)
    
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
        X = df.drop(['User ID', 'Purchased'], axis=1)   
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
    model_lr = LogisticRegression()
    model_nb = GaussianNB()
    model_svm = SVC()
    model_knn = KNeighborsClassifier()
    model_lr.fit(X_train, y_train)
    model_nb.fit(X_train, y_train)
    model_svm.fit(X_train, y_train)
    model_knn.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)
    y_pred_nb = model_nb.predict(X_test)
    y_pred_svm = model_svm.predict(X_test)
    y_pred_knn = model_knn.predict(X_test)

    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)

    precision_lr = precision_score(y_test, y_pred_lr)
    precision_nb = precision_score(y_test, y_pred_nb)
    precision_svm = precision_score(y_test, y_pred_svm)
    precision_knn = precision_score(y_test, y_pred_knn)

    recall_lr = recall_score(y_test, y_pred_lr)
    recall_nb = recall_score(y_test, y_pred_nb)
    recall_svm = recall_score(y_test, y_pred_svm)
    recall_knn = recall_score(y_test, y_pred_knn)

    f1_lr = f1_score(y_test, y_pred_lr)
    f1_nb = f1_score(y_test, y_pred_nb)
    f1_svm = f1_score(y_test, y_pred_svm)
    f1_knn = f1_score(y_test, y_pred_knn)

    results = pd.DataFrame({
        "Model": ["Logistic Regression", "Naive Bayes", "SVM", "KNN"],
        "Accuracy": [accuracy_lr, accuracy_nb, accuracy_svm, accuracy_knn],
        "Precision": [precision_lr, precision_nb, precision_svm, precision_knn],
        "Recall": [recall_lr, recall_nb, recall_svm, recall_knn],
        "F1 Score": [f1_lr, f1_nb, f1_svm, f1_knn]
    })

    print("\nModel Performance:")
    print(results.to_string(index=False))    

for d in ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]:
    run_dataset(d)