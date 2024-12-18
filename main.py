# Importing required modules

# Importing tools to work with data
import numpy as np
import pandas as pd

# Importing machine learning packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load dataset
df = pd.read_csv("final_preprocessed_dataset.csv")

# Preview of dataset
print("Preview of dataset:\n", df.head())

# Removing non numerical data from the dataset
df = df.drop('Unnamed: 0',1)
df = df.drop('Resource', 1)
df = df.drop('APT Group',1)

# Features
ylabels = np.array(df['APTGroup']) # APT group 
Xfeatures = df.drop('APTGroup', 1) # Features extracted from the malware samples

print("\nPreview of features dataset:\n", Xfeatures.head())

print("\nPreview of labels matrix:\n", ylabels)

# 70/30 split
x_train, x_test, y_train, y_test = train_test_split(Xfeatures, ylabels, test_size=0.2, random_state=42)

print("\nx_train shape:\n", x_train.head)

# Building the model
rfc = RandomForestClassifier(
    min_samples_leaf=50,
    n_estimators=150,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
    max_features='auto'
)

# Fit the training sets to the model
rfc.fit(x_train,y_train)

# Use model to make prediction
y_pred = rfc.predict(x_test)

print("\nPrediction results for test set:\n",y_pred)

# Accuracy of RFC model
print("\nAccuaracy of RFC model:\n", accuracy_score(y_test, y_pred))

# Cross validation using cross_val_score function
print("\nAccuracy of RFC model using cross validation:\n", cross_val_score(rfc, Xfeatures, ylabels, cv=20, scoring ='accuracy').mean())