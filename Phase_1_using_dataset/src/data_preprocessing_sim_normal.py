import pandas as pd
import numpy as np
from pathlib import Path

def preprocess_simulated_vanet(df, save_path):
    """
    Preprocess simulated VANET dataset for DDoS detection.
    Drops 'Confidence Normal' and 'Confidence DDoS'.
    Encodes IPs and Label, handles missing values, and saves as CSV.
    """
    # Drop Confidence columns if present
    df = df.drop(columns=[col for col in ['ConfidenceNormal', 'ConfidenceDDoS'] if col in df.columns])

    # Encode Label as binary
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Encode SourceIP and DestinationIP as categorical codes
    for col in ['SourceIP', 'DestinationIP']:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes

    # Optional: create inter-arrival time from Timestamp
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp').reset_index(drop=True)
        df['InterArrivalTime'] = df['Timestamp'].diff().fillna(0)

    # Handle missing / infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Create output directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Save preprocessed dataset as CSV
    df.to_csv(save_path, index=False)
    print(f"Preprocessed dataset saved to: {save_path}")
    print(f"Processed dataset shape: {df.shape}")
    print(f"Class distribution: {df['Label_Binary'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    # Load your simulated dataset CSV
    input_path = r"C:\Project\Phase_1_using_dataset\data\raw\Simulated_Dataset\vanet-ddos-trace.csv" 
    df_raw = pd.read_csv(input_path)

    # Set output path
    output_path = "C:\Project\Phase_1_using_dataset\data\processed\sim_ddos_processed.csv"

    # Preprocess and save
    df_processed = preprocess_simulated_vanet(df_raw, output_path)



