# ctgan_training_vanet_tqdm.py
import pandas as pd
from pathlib import Path
import joblib
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from tqdm import trange

def load_data(normal_path, ddos_path):
    """Load and merge VANET normal and DDoS datasets"""
    print("Loading VANET datasets for CTGAN training...")
    df_normal = pd.read_csv(normal_path)
    df_ddos = pd.read_csv(ddos_path)

    df = pd.concat([df_normal, df_ddos], ignore_index=True)
    print(f"Combined dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Label_Binary'].value_counts()}")
    return df

def train_ctgan(df, model_save_path, epochs=100, batch_size=500):
    """Train CTGAN with a tqdm progress bar for epochs"""
    print("\nTraining CTGAN model...")

    # Create metadata automatically
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Initialize CTGAN
    ctgan = CTGANSynthesizer(metadata=metadata, epochs=1, batch_size=batch_size, cuda=True, verbose=False)

    # Use tqdm to display epoch progress
    for epoch in trange(epochs, desc="CTGAN Epochs", unit="epoch"):
        ctgan.fit(df)

    # Save trained model
    joblib.dump(ctgan, str(model_save_path))
    print(f"\n✅ CTGAN model saved to: {model_save_path}")
    return ctgan

def generate_augmented_data(ctgan, n_samples, save_path):
    """Generate synthetic dataset using trained CTGAN"""
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic_data = ctgan.sample(n_samples)
    # Convert Path to string
    synthetic_data.to_csv(str(save_path), index=False)
    print(f"✅ Augmented dataset saved to: {save_path}")
    return synthetic_data


def main():
    normal_path = r"C:\Project\Phase_1_using_dataset\data\processed\processed_simulated_dataset\sim_normal_processed.csv"
    ddos_path   = r"C:\Project\Phase_1_using_dataset\data\processed\processed_simulated_dataset\sim_ddos_processed.csv"
    results_dir = Path(r"C:\Project\Phase_1_using_dataset\data\extented_augmented_data")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(normal_path, ddos_path)

    # File paths
    model_save_path = results_dir / "extended_ctgan_vanet_model.pkl"
    augmented_save_path = results_dir / "extended_vanet_ctgan_augmented.csv"

    # Train CTGAN with epoch progress bar
    ctgan = train_ctgan(df, model_save_path, epochs=100, batch_size=500)

    # Generate synthetic dataset
    generate_augmented_data(ctgan, n_samples=100000, save_path=augmented_save_path)

    print("\n CTGAN training and augmentation complete!")

if __name__ == "__main__":
    main()
