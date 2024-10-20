# supervised_classification.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the dataset
# Replace 'your_dataset.csv' with the path to your dataset file
data = pd.read_csv('stroke_data.csv')

# Define features and target variable
X = data.drop(columns=['stroke'])
y = data['stroke']

# Preprocessing: One-hot encoding for categorical variables
categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Random Forest Classifier
rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', RandomForestClassifier(random_state=42))])
rf_clf.fit(X_train, y_train)

# Predictions and evaluation for Random Forest
rf_preds = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])

print(f'Random Forest Classifier - Accuracy: {rf_accuracy * 100:.2f}%')
print(f'Random Forest Classifier - AUC: {rf_auc:.2f}')

# AdaBoost Classifier
ada_clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', AdaBoostClassifier(random_state=42))])
ada_clf.fit(X_train, y_train)

# Predictions and evaluation for AdaBoost
ada_preds = ada_clf.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_preds)
ada_auc = roc_auc_score(y_test, ada_clf.predict_proba(X_test)[:, 1])

print(f'AdaBoost Classifier - Accuracy: {ada_accuracy * 100:.2f}%')
print(f'AdaBoost Classifier - AUC: {ada_auc:.2f}')