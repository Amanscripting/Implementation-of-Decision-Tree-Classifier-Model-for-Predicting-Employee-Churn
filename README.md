# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries and load the dataset.

2.Handle null values and encode categorical columns.

3.Split data into training and testing sets.

4.Train a DecisionTreeClassifier using entropy.

5.Predict and evaluate the model using accuracy and metrics.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: AMAN ALAM
RegisterNumber:  212224240011

from google.colab import drive
drive.mount('/content/drive')
from google.colab import files
uploaded = files.upload()

import pandas as pd
import io

data = pd.read_csv(io.BytesIO(uploaded['Employee.csv']))
data.head()


data.head()
data.info()
print("Null values:\n", data.isnull().sum())
print("Class distribution:\n", data["left"].value_counts())
from sklearn.preprocessing import LabelEncoder

# Encode categorical features
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Display the updated data
data.head()
# Select input features
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours",
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
print(x.head())

# Define target variable
y = data["left"]

from sklearn.model_selection import train_test_split

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
from sklearn.tree import DecisionTreeClassifier

# Create and train the model
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train, y_train)
from sklearn import metrics

# Predict on test set
y_pred = dt.predict(x_test)

# Evaluate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Predict on new employee data
sample_prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print("Sample Prediction:", sample_prediction)


*/
```

## Output:
![image](https://github.com/user-attachments/assets/371f65c6-d860-48f6-a2f6-67abf05d34d1)

![image](https://github.com/user-attachments/assets/c5e17d4d-69e8-43c7-b5be-c702c1bb5c00)

![image](https://github.com/user-attachments/assets/0ec1cc80-070e-454e-ad35-2e81007ae512)

![image](https://github.com/user-attachments/assets/f96e6837-8acc-425f-88f0-28c2d0bdb1c4)

![image](https://github.com/user-attachments/assets/9908a55f-d37f-4ac6-a4b6-61953ab22ffc)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
