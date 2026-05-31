import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def l1_distance(a, b):
    return np.sum(np.abs(a - b))
def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def dtw_distance(x, y):
    n, m = len(x), len(y)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]
def knn_predict(X_train, y_train, X_test, k, dist_func):
    preds = []
    for tp in X_test:
        distances = []
        for i in range(len(X_train)):
            d = dist_func(tp, X_train[i])
            distances.append((d, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        labels = [label for _, label in neighbors]
        preds.append(max(set(labels), key=labels.count))
    return np.array(preds)
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
        # for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        #     df[col] = df[col].replace(0, df[col].median())

        # X = df.drop('Outcome', axis=1).values
        # y = df['Outcome'].values
        df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
        X = df[["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]
        y = df['Outcome'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    k = int(input("Enter value of k : "))

    y_l1 = knn_predict(X_train, y_train, X_test, k, l1_distance)
    y_l2 = knn_predict(X_train, y_train, X_test, k, l2_distance)
    y_dtw = knn_predict(X_train, y_train, X_test, k, dtw_distance)

    acc_l1 = (y_l1 == y_test).mean()
    acc_l2 = (y_l2 == y_test).mean()
    acc_dtw = (y_dtw == y_test).mean()

    print("L1 Accuracy  :", acc_l1)
    print("L2 Accuracy  :", acc_l2)
    print("DTW Accuracy :", acc_dtw)

datasets = ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]

for d in datasets:
    run_dataset(d)