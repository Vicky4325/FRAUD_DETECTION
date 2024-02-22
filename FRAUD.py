# Importing necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("/kaggle/input/fraud-detection/fraudTest.csv", encoding='ISO-8859-1')

# Preprocessing
data.dropna(subset=['unix_time', 'merch_lat', 'merch_long', 'is_fraud'], inplace=True)
data.fillna({'city_pop': data['city_pop'].median()}, inplace=True)

# Encoding categorical variables
categorical_cols = ['Unnamed: 0', 'cc_num', 'category', 'trans_date_trans_time', 'amt', 
                    'merchant', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 
                    'merch_lat', 'merch_long', 'first', 'last', 'street', 
                    'job', 'dob', 'trans_num', 'gender', 'city', 'state']
data[categorical_cols] = data[categorical_cols].apply(lambda x: pd.factorize(x)[0])

# Splitting the dataset into features and target variable
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Training the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Results
print("Decision Tree Model:")
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
