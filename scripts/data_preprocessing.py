# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def load_data(file_path):
    """Loads the dataset from the given file path."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Cleans and preprocesses the data."""
    
    # Fill missing BMI values with the mean
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['gender', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
    
    # Balancing the dataset by resampling
    positive_cases = df[df['stroke'] == 1]
    negative_cases = df[df['stroke'] == 0]
    
    negative_downsampled = resample(negative_cases, replace=False, n_samples=len(positive_cases), random_state=42)
    
    df_balanced = pd.concat([positive_cases, negative_downsampled])
    
    # Separating features and labels
    X = df_balanced.drop(['stroke', 'id'], axis=1)
    y = df_balanced['stroke']
    
    return X, y

def split_data(X, y, test_size=0.3):
    """Splits the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Example usage
    file_path = 'data/stroke_data.csv'
    df = load_data(file_path)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Data preprocessed and split successfully!")