# train_baseline.py
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
    """Load the preprocessed dataset"""
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Label_Binary'].value_counts()}")
    return df

def prepare_data(df):
    """Prepare features and target variable"""
    # Separate features and target
    X = df.drop('Label_Binary', axis=1)
    y = df['Label_Binary']
    
    # Split the data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def find_top_features(model, X_train, top_n=20):
    """Identify the most important features"""
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"Top {top_n} most important features:")
    for i, (feature, importance) in enumerate(feature_importance.head(top_n).values, 1):
        print(f"{i:2d}. {feature}: {importance:.6f}")
    
    return feature_importance.head(top_n)['feature'].tolist()

def train_with_limited_features(X_train, y_train, X_test, y_test, n_features=10):
    """Train model using only top n features"""
    # First train full model to find important features
    full_model = RandomForestClassifier(n_estimators=50, random_state=42)
    full_model.fit(X_train, y_train)
    
    # Get top features
    top_features = find_top_features(full_model, X_train, n_features)
    
    # Train new model with only top features
    X_train_limited = X_train[top_features]
    X_test_limited = X_test[top_features]
    
    limited_model = RandomForestClassifier(n_estimators=100, random_state=42)
    limited_model.fit(X_train_limited, y_train)
    
    # Evaluate
    y_pred_limited = limited_model.predict(X_test_limited)
    accuracy_limited = accuracy_score(y_test, y_pred_limited)
    
    print(f"\nAccuracy with {n_features} features: {accuracy_limited:.4f}")
    return limited_model, top_features, accuracy_limited

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest model...")
    
    # Initialize Random Forest with optimal parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Training completed!")
    return rf_model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    return y_pred, accuracy, precision, recall, f1

def plot_confusion_matrix(y_test, y_pred, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Random Forest Benchmark')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot and save feature importance - FIXED VERSION"""
    importance = model.feature_importances_
    
    # If we have fewer features than top_n, adjust
    top_n = min(top_n, len(importance))
    
    indices = np.argsort(importance)[::-1][:top_n]  # Get top N indices
    
    plt.figure(figsize=(12, 8))
    plt.title(f"Feature Importance - Top {top_n} Features")
    plt.bar(range(top_n), importance[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in range(top_n)], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def save_results(metrics, save_path):
    """Save evaluation metrics to file"""
    with open(save_path, 'w') as f:
        f.write("Random Forest Benchmark Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")

def main():
    # Configuration
    processed_data_path = "C:\Project\Phase_1_using_dataset\data\processed\cic_ids2017_processed.csv"
    results_dir = "C:/Project/Phase_1_using_dataset/results/ids2017"
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load processed data
    df = load_processed_data(processed_data_path)
    
    # Step 2: Prepare data for training
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Step 3: Train with LIMITED FEATURES (10 features)
    print("\n" + "="*60)
    print("TRAINING WITH LIMITED FEATURES (10 features)")
    print("="*60)
    
    limited_model, top_features, accuracy_limited = train_with_limited_features(
        X_train, y_train, X_test, y_test, n_features=10
    )
    
    # Step 4: Full evaluation of the limited model
    y_pred, accuracy, precision, recall, f1 = evaluate_model(limited_model, X_test[top_features], y_test)
    
    # Step 5: Save plots and results
    plot_confusion_matrix(y_test, y_pred, f"{results_dir}/confusion_matrix_10features.png")
    plot_feature_importance(limited_model, top_features, f"{results_dir}/feature_importance_10features.png")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'features_used': top_features
    }
    save_results(metrics, f"{results_dir}/metrics_10features.txt")
    
    # Save the trained model
    joblib.dump(limited_model, f"{results_dir}/baseline_random_forest_10features.pkl")
    print(f"\nModel saved to: {results_dir}/baseline_random_forest_10features.pkl")
    
    print("\nPhase 1 Benchmark Complete with 10 features!")
    print(f"Results saved in: {results_dir}")

if __name__ == "__main__":
    main()