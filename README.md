# Predictive-Modeling-Using-Machine-Learning
Build a model to predict outcomes based on given data.

Title
Customer Purchase Prediction Using Machine Learning

Objective
The objective of this project was to build a machine learning model that predicts customer purchase behavior based on historical data.

Tools Used
Python
Pandas
Scikit-learn
Matplotlib
Seaborn

Dataset Features
Age
Income
Gender
Previous Purchases
Product Category

Steps Performed

1. Data Preprocessing
Removed missing values
Encoded categorical variables
Split the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

2. Model Building

Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

3. Prediction
predictions = model.predict(X_test)

4. Model Evaluation
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

Results
Model Accuracy: 87%
The model successfully predicted customer purchase decisions based on demographic and historical information.

Conclusion
This project provided practical experience in supervised machine learning. I learned how to preprocess data, train predictive models, and evaluate their performance using accuracy metrics and confusion matrices.
