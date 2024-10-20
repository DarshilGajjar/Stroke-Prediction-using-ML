# evaluate_model.py

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

def load_model(filename):
    """Loads a trained model from a file."""
    return joblib.load(filename)

def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    report = classification_report(y_test, y_pred)
    
    return accuracy, roc_auc, report

if __name__ == "__main__":
    from data_preprocessing import X_test, y_test
    
    # Load trained models
    rf_model = load_model('models/random_forest_model.pkl')
    ada_model = load_model('models/adaboost_model.pkl')
    
    # Evaluate Random Forest model
    rf_accuracy, rf_roc_auc, rf_report = evaluate_model(rf_model, X_test, y_test)
    print("Random Forest Model Performance:")
    print(f"Accuracy: {rf_accuracy}")
    print(f"ROC AUC: {rf_roc_auc}")
    print(rf_report)
    
    # Evaluate AdaBoost model
    ada_accuracy, ada_roc_auc, ada_report = evaluate_model(ada_model, X_test, y_test)
    print("\nAdaBoost Model Performance:")
    print(f"Accuracy: {ada_accuracy}")
    print(f"ROC AUC: {ada_roc_auc}")
    print(ada_report)