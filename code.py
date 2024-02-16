//Mounting The drive and loading the dataset
from google.colab import drive
drive.mount('/content/drive')
credit_card_data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
// Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
//dataset informations
credit_card_data.describe()
credit_card_data.info()
credit_card_data.head(5)
credit_card_data.tail(5)
//checking the number of missing values in each column
credit_card_data.isnull().sum()
// Class Imbalance Check 
print(credit_card_data['Class'].value_counts())
// Handle Imbalance 
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=11)
x_smote, y_smote = smote.fit_resample(X_train, y_train)
print("Shape before the Oversampling : ",X_train.shape)
print("Shape after the Oversampling : ",x_smote.shape)
//data pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = credit_card_data.drop('Class', axis=1)
Y = credit_card_data.Class
X = scalar.fit_transform(X)
//separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]
credit_card_data['Class'].value_counts()
legit.Amount.describe()
fraud.Amount.describe()
credit_card_data.groupby('Class').mean()
// Data Visualization
// Scatter plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(legit['Time'], legit['Amount'], color='blue', label='Legitimate')
plt.scatter(fraud['Time'], fraud['Amount'], color='red', label='Fraudulent')
plt.title('Scatter Plot of Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend()
plt.show()
// Box plot
import seaborn as sns
combined_data = pd.concat([legit, fraud])
plt.figure(figsize=(10, 6))
sns.boxplot(x='Class', y='Amount', data=combined_data)
plt.title('Distribution of Transaction Amount by Class')
plt.xlabel('Class (0: Legitimate, 1: Fraudulent)')
plt.ylabel('Transaction Amount')
plt.show()
//splitting the data into testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
//Building Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
}
best_accuracy = 0
best_model = None
best_model_name = None
results = {}
for model_name, model in models.items():
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    if accuracy>best_accuracy:
      best_accuracy = accuracy
      best_model_name = model_name
      best_model = model
results_df = pd.DataFrame(results).T
print(results_df)
y_pred_test = best_model.predict(X_test)
test_accuracy = accuracy_score(Y_test, y_pred_test)
print(f"Test accuracy for the best model({best_model_name}): {test_accuracy*100}")
