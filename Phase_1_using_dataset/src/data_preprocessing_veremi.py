import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_veremi(df, save_path):
    """
    Preprocess VeReMi extension VANET dataset for DDoS detection.
    Encodes categorical features, derives inter-arrival time, and ensures all numeric features.
    """
    # Encode Label as binary: 0 = Normal, 1 = Attack
    if 'Attack' in df.columns:
        df['Label_Binary'] = df['Attack'].apply(lambda x: 0 if x == 0 else 1)

    # Encode categorical/text columns as numeric codes
    categorical_cols = ['type', 'sender', 'senderPseudo', 'class', 'Attack_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Optional: derive inter-arrival time from sendTime
    if 'sendTime' in df.columns:
        df = df.sort_values('sendTime').reset_index(drop=True)
        df['InterArrivalTime'] = df['sendTime'].diff().fillna(0)

    # Handle missing / infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed dataset
    df.to_csv(save_path, index=False)
    print(f"Preprocessed VeReMi dataset saved to: {save_path}")
    print(f"Processed dataset shape: {df.shape}")
    print(f"Class distribution: {df['Label_Binary'].value_counts().to_dict()}")

    return df

if __name__ == "__main__":
    # Load VeReMi extension dataset
    input_path = r"C:\Project\Phase_1_using_dataset\data\raw\Veremi\veremi_extension_simple.csv"
    df_raw = pd.read_csv(input_path)

    # Set output path
    output_path = r"C:\Project\Phase_1_using_dataset\data\processed\veremi_processed.csv"

    # Preprocess and save
    df_processed = preprocess_veremi(df_raw, output_path)
