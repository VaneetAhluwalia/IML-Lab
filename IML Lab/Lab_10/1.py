import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def run_dataset(file_name):
    print("\nDataset :", file_name)
    df = pd.read_csv(file_name)
    
    if file_name == "titanic.csv":
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df = df.drop(columns=['Cabin', 'Sex', 'Name', 'Ticket', 'Embarked', 'PassengerId'])
        X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]]

    elif file_name == "Social_Network_Ads.csv":
        df = df.drop(columns=['Gender'])
        X = df.drop(['User ID', 'Purchased'], axis=1)

    elif file_name == "diabetes_dataset.csv":
        df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                      "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age"]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    model = BisectingKMeans(n_clusters=2, random_state=42)
    model.fit(X)
    labels = model.labels_

    score = silhouette_score(X, labels)
    print("Silhouette Score:", score)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title(f"Clusters for {file_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


for d in ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]:
    run_dataset(d)