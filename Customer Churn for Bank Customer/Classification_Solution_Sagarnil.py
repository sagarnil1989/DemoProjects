#Evaluating, Improving, Tuning Artificial Neural Network

# Part 1 - Data Preprocessing
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Set working directory & import data
dataset = pd.read_csv("Churn_Modelling.csv")
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#To avoid dummy variable trap, we have removed the first column of X
X = X[:, 1:]

# Spliting the dataset into Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling (to make the features on same scale)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_LogicalRegression = confusion_matrix(Y_test, Y_pred)

#-------------------
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier= SVC(kernel='linear', random_state=0)
classifier.fit(X_train,Y_train)
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm_SVM = confusion_matrix(Y_test, Y_pred)

#---------------------
# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier= SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,Y_train)
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm_SVM_Kernel = confusion_matrix(Y_test, Y_pred)

#------------------
# Fitting Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
Y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm_RandomForest = confusion_matrix(Y_test, Y_pred)

