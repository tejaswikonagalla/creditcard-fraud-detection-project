# creditcard-fraud-detection-project
The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud.
So,this system is used to identify whether a new transaction is fraudulent or not. 
Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.
This project aims to develop a credit card fraud detection system using machine learning techniques.
The goal is to create a model that can accurately identify fraudulent transactions.

## Dataset
 The dataset is collected from "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud".
 It has 2,84,807 transaction details out of which 492 are fraud.
 
 ## Features
  Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'.
  Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
  The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning.
  Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

  ## Dependencies
  -Python 3.7 <br /> 
  -numpy  <br />
  -pandas 1.2.4  <br />
  -matplotlib 3.3.4  <br />
  -sklearn.model_selection <br />
  -sklearn.linear_model <br />
  -sklearn.metrics <br />
  -sklearn.ensemble <br />
  -sklearn.tree <br />
  -imblearn.over_sampling <br />
  -seaborn <br />
  -sklearn.svm <br />
  -gradio

  ## Installations
  pip install gradio

  ## Models used
  Logistic Regression <br />
  Decision tree <br />
  Random forest <br />
  SVM
  
  ## Conclusion
  Random Forest is the most accurate model for credit card fraud detection.
