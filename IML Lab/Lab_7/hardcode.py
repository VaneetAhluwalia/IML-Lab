import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
class NB_HC:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.priors = {}

        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(
                -0.5 * np.log(2 * np.pi * self.var[c])
                - ((x - self.mean[c]) ** 2) / (2 * self.var[c])
            )
            posteriors.append(prior + likelihood)

        return self.classes[np.argmax(posteriors)]

class SVM_HC:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000): 
        self.lr = lr 
        self.lambda_param = lambda_param 
        self.n_iters = n_iters 
    def fit(self, X, y): 
        y_ = np.where(y <= 0, -1, 1) 
        n_samples, n_features = X.shape 
        self.w = np.zeros(n_features) 
        self.b = 0 
        for _ in range(self.n_iters): 
            for idx, x_i in enumerate(X): 
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1 
                if condition: 
                    self.w -= self.lr * (2 * self.lambda_param * self.w) 
                else: 
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])) 
                    self.b -= self.lr * y_[idx] 
    def predict(self, X): 
        linear_output = np.dot(X, self.w) - self.b 
        return np.where(linear_output >= 0, 1, 0)

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
        X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
        y = df['Survived'].values

    elif file_name == "Social_Network_Ads.csv":
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

        X = df.drop(['User ID', 'Purchased'], axis=1)
        y = df['Purchased'].values

    elif file_name == "diabetes_dataset.csv":
        df.columns = [
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
        ]

        X = df[[
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"
        ]]

        y = df['Outcome'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_nb = NB_HC()
    model_nb.fit(X_train, y_train)
    y_pred_nb = model_nb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

    model_svm = SVM_HC()
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

datasets = ["diabetes_dataset.csv", "Social_Network_Ads.csv", "titanic.csv"]

for d in datasets:
    run_dataset(d)