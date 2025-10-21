import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False


# ==============================================================
# DATA LOADING
# ==============================================================

def load_processed_data(normal_path, ddos_path):
    print("Loading processed VANET datasets...")
    df_normal = pd.read_csv(normal_path)
    df_ddos = pd.read_csv(ddos_path)

    df = pd.concat([df_normal, df_ddos], ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Label_Binary'].value_counts()}")

    return df


# ==============================================================
# DATA PREPARATION
# ==============================================================

def prepare_data(df):
    """Drop non-numeric columns and split features/target."""
    X = df.drop(['Label_Binary', 'Label'], axis=1, errors='ignore')
    y = df['Label_Binary']

    # Ensure all are numeric
    X = X.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==============================================================
# MODEL TRAINING & EVALUATION
# ==============================================================

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred)
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    return metrics, y_pred


def train_and_evaluate_models(X_train, y_train, X_test, y_test, results_dir):
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'NaiveBayes': GaussianNB()
    }

    if xgb_available:
        models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    results = []

    for name, model in models.items():
        print("\n" + "=" * 60)
        print(f"Training {name} model...")
        model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f"{results_dir}/confusion_matrix_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save model
        joblib.dump(model, f"{results_dir}/{name}_model.pkl")

    return pd.DataFrame(results)


# ==============================================================
# MAIN SCRIPT
# ==============================================================

def main():

    normal_path = r"C:\Project\Phase_1_using_dataset\data\processed\processed_simulated_dataset\sim_normal_processed.csv"
    ddos_path   = r"C:\Project\Phase_1_using_dataset\data\processed\processed_simulated_dataset\sim_ddos_processed.csv"
    results_dir = Path(r"C:\Project\Phase_1_using_dataset\results\sim_dataset_comparison")

    results_dir.mkdir(parents=True, exist_ok=True)


    # Load data
    df = load_processed_data(normal_path, ddos_path)

    # Prepare
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train & compare
    results_df = train_and_evaluate_models(X_train, y_train, X_test, y_test, results_dir)

    # Save results summary
    results_path = results_dir / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ All model results saved to: {results_path}")

    print("\nModel performance summary:")
    print(results_df.sort_values('F1', ascending=False))


if __name__ == "__main__":
    main()
