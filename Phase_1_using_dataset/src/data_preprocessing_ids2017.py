import pandas as pd
import numpy as np
from pathlib import Path

def load_cic_ids2017(data_dir):
    """
    Load CIC-IDS2017 dataset - focus on DDoS attacks from Friday data
    """
    data_path = Path(data_dir)
    
    # The Friday file contains DDoS attacks
    friday_file = data_path / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    
    print(f"Loading CIC-IDS2017 data from: {friday_file}")
    
    # Load the data
    df = pd.read_csv(friday_file, encoding='utf-8', low_memory=False)
    
    print(f"Original dataset shape: {df.shape}")
    return df

def preprocess_cic_ids2017(df):
    """
    Preprocess CIC-IDS2017 data for DDoS detection with comprehensive features
    """
    print("\nPreprocessing CIC-IDS2017 data...")
    
    # Use the correct label column (with space)
    label_col = ' Label'
    
    # Create binary labels
    df['Label_Binary'] = df[label_col].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # COMPREHENSIVE FEATURE SET - All potentially useful features
    all_potential_features = [
        # Basic flow characteristics
        ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
        'Total Length of Fwd Packets', ' Total Length of Bwd Packets',
        
        # Packet size statistics
        ' Fwd Packet Length Max', ' Fwd Packet Length Min', 
        ' Fwd Packet Length Mean', ' Fwd Packet Length Std',
        'Bwd Packet Length Max', ' Bwd Packet Length Min', 
        ' Bwd Packet Length Mean', ' Bwd Packet Length Std',
        
        # Rate-based features (CRITICAL for DDoS)
        'Flow Bytes/s', ' Flow Packets/s', 
        'Fwd Packets/s', ' Bwd Packets/s',
        
        # Inter-Arrival Time features (VERY important for attack detection)
        ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
        ' Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min',
        'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min',
        
        # TCP flags (ESSENTIAL for SYN floods and other TCP-based attacks)
        'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', 
        ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
        
        # Protocol and port information
        ' Destination Port', ' Protocol',
        
        # Header information
        ' Fwd Header Length', ' Bwd Header Length',
        
        # Bulk transfer characteristics
        'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
        ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
        
        # Window size and segment information
        'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
        ' min_seg_size_forward',
        
        # Active/Idle times
        'Active Mean', ' Active Std', ' Active Max', ' Active Min',
        'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
        
        # Binary label for target
        'Label_Binary'
    ]
    
    # Keep only features that actually exist in the dataset
    available_features = [f for f in all_potential_features if f in df.columns]
    print(f"Using {len(available_features)} available features")
    
    df_processed = df[available_features].copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
    
    # Encode protocol (categorical feature)
    if ' Protocol' in df_processed.columns:
        df_processed[' Protocol'] = df_processed[' Protocol'].astype('category').cat.codes
    
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Class distribution: {df_processed['Label_Binary'].value_counts().to_dict()}")

    # Remove infinite values and extreme outliers
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    # Replace inf with NaN
    df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN (or fill with median)
    df_processed = df_processed.dropna()
    
    print(f"Final cleaned dataset shape: {df_processed.shape}")
    return df_processed
    
    return df_processed

def main():
    # Configuration
    raw_data_dir = "C:/Project/Phase_1_using_dataset/data/raw/CIC-2017"
    output_path = "C:\Project\Phase_1_using_dataset\data\processed\cic_ids2017_processed.csv"
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting CIC-IDS2017 DDoS data preprocessing...")
    
    # Step 1: Load data
    df = load_cic_ids2017(raw_data_dir)
    
    # Step 2: Preprocess
    df_processed = preprocess_cic_ids2017(df)
    
    # Step 3: Save processed data
    df_processed.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    return df_processed

if __name__ == "__main__":
    df = main()

    #new comment