# model_development.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

def train_model():
    # 1. Load the dataset
    # Download 'data.csv' from Kaggle (Breast Cancer Wisconsin Diagnostic)
    try:
        df = pd.read_csv('data.csv')
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: 'data.csv' not found. Please ensure the dataset is in the folder.")
        return

    # 2. Feature Selection
    # We select 5 specific features + the target
    feature_cols = ['radius_mean', 'texture_mean', 'smoothness_mean', 
                   'compactness_mean', 'symmetry_mean']
    target_col = 'diagnosis'
    
    X = df[feature_cols]
    y = df[target_col]

    # 3. Data Preprocessing
    print("Preprocessing data...")
    
    # a. Encoding categorical target (M = Malignant, B = Benign)
    # We map Malignant to 1 and Benign to 0
    y = y.map({'M': 1, 'B': 0})
    
    # b. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # c. Feature Scaling (MANDATORY for SVM)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Implementation: Support Vector Machine (SVM)
    print("Training SVM Model...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)

    # 5. Evaluation
    y_pred = model.predict(X_test_scaled)
    
    print("\n--- Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

    # 6. Save Model AND Scaler
    # We must save the scaler to transform user input in the app later!
    joblib.dump(model, 'cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\nFiles saved: 'cancer_model.pkl' and 'scaler.pkl'")

if __name__ == "__main__":
    train_model()