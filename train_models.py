import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load existing training data from sample_training_data.csv
def load_training_data():
    try:
        # Load the existing training data
        df = pd.read_csv('models/sample_training_data.csv')
        print(f"Successfully loaded existing training data with {len(df)} samples.")
        
        # Check if required columns exist
        required_columns = ['Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves', 'Selected']
        if not all(col in df.columns for col in required_columns):
            print("Error: Required columns not found in training data.")
            print(f"Found columns: {list(df.columns)}")
            print("Please ensure your CSV has: Goals, Matches, Assists, Pass Accuracy, Tackles, Saves, Selected")
            return None
        
        return df
        
    except FileNotFoundError:
        print("Error: sample_training_data.csv not found in models folder.")
        return None
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

# Train models
def train_models(df):
    if df is None:
        print("Cannot train models: No training data available.")
        return
    print(f"Training models on {len(df)} samples...")
    # Split features and target
    X = df[['Goals', 'Matches', 'Assists', 'Pass Accuracy', 'Tackles', 'Saves']].values
    y = df['Selected'].values
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Train Logistic Regression model
    lr_model = LogisticRegression(
        random_state=42, 
        max_iter=1000,
        C=1.0,
        penalty='l2'
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    # Train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    

    
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Check class balance in training data
    print(f"\nTraining data class balance: {np.mean(y_train):.3f}")
    
    # Test probability distributions
    print("\nTesting probability distributions...")
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    print(f"LR Probabilities - Min: {lr_probs.min():.3f}, Max: {lr_probs.max():.3f}, Mean: {lr_probs.mean():.3f}")
    print(f"RF Probabilities - Min: {rf_probs.min():.3f}, Max: {rf_probs.max():.3f}, Mean: {rf_probs.mean():.3f}")
    # Check how many would be selected with 0.75 threshold (new selection criteria)
    lr_selected = np.mean(lr_probs >= 0.75)
    rf_selected = np.mean(rf_probs >= 0.75)
    avg_probs = (lr_probs + rf_probs) / 2
    avg_selected = np.mean(avg_probs >= 0.75)
    print(f"\nSelection rates with 0.75 threshold:")
    print(f"LR: {lr_selected:.3f}, RF: {rf_selected:.3f}, Average: {avg_selected:.3f}")# Save models and scaler
    with open('models/logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Models and scaler saved to 'models/' directory.")# Save a backup of the training data used
    df.to_csv('models/training_data_used.csv', index=False)
    print("Training data used saved to 'models/training_data_used.csv'.")# Main execution
if __name__ == "__main__":
    print("Loading training data from models/sample_training_data.csv...")
    df = load_training_data()
    if df is not None:
        print(f"Using {len(df)} samples for training.")
        print(f"Data columns: {list(df.columns)}")
        print("Note: No feature weightage applied - models learn directly from data patterns.")
        print("\nTraining models...")
        train_models(df)
        print("\nDone! You can now run the Flask application.")
        print("Selection criteria: Average probability >= 0.75")
    else:
        print("Training failed. Please check your training data file.")
