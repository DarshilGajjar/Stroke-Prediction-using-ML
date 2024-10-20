# train_model.py

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier."""
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_adaboost(X_train, y_train):
    """Trains an AdaBoost classifier."""
    ada = AdaBoostClassifier(random_state=42)
    ada.fit(X_train, y_train)
    return ada

def save_model(model, filename):
    """Saves the trained model to a file."""
    joblib.dump(model, filename)

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import X_train, y_train
    
    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)
    save_model(rf_model, 'models/random_forest_model.pkl')
    
    # Train AdaBoost model
    ada_model = train_adaboost(X_train, y_train)
    save_model(ada_model, 'models/adaboost_model.pkl')
    
    print("Models trained and saved successfully!")