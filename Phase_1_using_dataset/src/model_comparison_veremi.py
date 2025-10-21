# train_baseline_veremi_clean.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

def load_processed_data(data_path):
    print("Loading Veremi processed data...")
    df = pd.read_csv(data_path)
    
    # Keep only numeric columns and Label_Binary
    df = df.select_dtypes(include=[np.number])
    
    # Ensure Label_Binary exists
    if 'Label_Binary' not in df.columns:
        raise ValueError("No column for labels found in dataset!")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Label_Binary'].value_counts()}")
    return df

def prepare_data(df):
    # Drop Label_Binary from features
    X = df.drop('Label_Binary', axis=1)
    y = df['Label_Binary']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_with_limited_features(X_train, y_train, X_test, y_test, n_features=10):
    # Train full model to get top features
    full_model = RandomForestClassifier(n_estimators=50, random_state=42)
    full_model.fit(X_train, y_train)
    
    # Top features
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': full_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(n_features)['feature'].tolist()
    
    X_train_limited = X_train[top_features]
    X_test_limited = X_test[top_features]
    
    limited_model = RandomForestClassifier(n_estimators=100, random_state=42)
    limited_model.fit(X_train_limited, y_train)
    
    y_pred = limited_model.predict(X_test_limited)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy with {n_features} features: {accuracy:.4f}")
    
    return limited_model, top_features, accuracy

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Only print report if more than 1 class exists
    if len(np.unique(y_test)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    else:
        print("\nOnly one class present. Skipping classification report.")
    
    return y_pred, accuracy, precision, recall, f1

def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest (Veremi)')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    processed_data_path = r"C:\Project\Phase_1_using_dataset\data\processed\veremi_processed.csv"
    results_dir = Path(r"C:\Project\Phase_1_using_dataset\results\veremi_clean")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_processed_data(processed_data_path)
    
    # Skip if only one class exists
    if len(np.unique(df['Label_Binary'])) < 2:
        print("Only one class present. Skipping training.")
        return
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    limited_model, top_features, acc_limited = train_with_limited_features(
        X_train, y_train, X_test, y_test, n_features=10
    )
    
    y_pred, accuracy, precision, recall, f1 = evaluate_model(
        limited_model, X_test[top_features], y_test
    )
    
    # Save confusion matrix and model
    plot_confusion_matrix(y_test, y_pred, f"{results_dir}/confusion_matrix_veremi.png")
    joblib.dump(limited_model, f"{results_dir}/rf_veremi_baseline_clean.pkl")
    print(f"\nModel saved to: {results_dir}/rf_veremi_baseline_clean.pkl")

if __name__ == "__main__":
    main()
