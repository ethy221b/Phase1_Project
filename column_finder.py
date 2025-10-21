# analyze_columns.py
import pandas as pd
from pathlib import Path

def analyze_dataset_structure(main_data_dir):
    """
    Analyze column structure across both folders (01-12 and 03-11)
    """
    main_path = Path(main_data_dir)
    
    # Define the two folder paths
    folder1_path = main_path / "01-12"
    folder2_path = main_path / "03-11"
    
    print(f"Analyzing dataset structure in: {main_data_dir}")
    print("=" * 60)
    
    # Get all CSV files from both folders
    all_files = []
    all_files.extend(folder1_path.glob("*.csv"))
    all_files.extend(folder2_path.glob("*.csv"))
    
    print(f"Found {len(all_files)} CSV files across both folders")
    
    column_analysis = {}
    
    for i, file in enumerate(all_files[:18]): 
        print(f"\nAnalyzing {file.parent.name}/{file.name}:")
        
        try:
            # Read just the header to get column names
            df_sample = pd.read_csv(file, nrows=2, encoding='utf-8', 
                                  on_bad_lines='skip', low_memory=False)
            
            column_analysis[f"{file.parent.name}/{file.name}"] = {
                'column_count': len(df_sample.columns),
                'columns': list(df_sample.columns),
                'label_column': 'Label' if 'Label' in df_sample.columns else None
            }
            
            print(f"  Columns: {len(df_sample.columns)}")
            print(f"  Label present: {'Label' in df_sample.columns}")
            print(f"  First 10 columns: {list(df_sample.columns)[:10]}")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return column_analysis

def find_common_columns(column_analysis):
    """
    Find columns that exist in all files across both folders
    """
    all_columns = []
    for file_info in column_analysis.values():
        all_columns.extend(file_info['columns'])
    
    # Get unique columns
    unique_columns = set(all_columns)
    
    print(f"\nFound {len(unique_columns)} unique columns across all files")
    print("All unique columns:", sorted(unique_columns))
    
    # Find columns common to all analyzed files
    common_columns = set()
    first_file = True
    
    for file_info in column_analysis.values():
        if first_file:
            common_columns = set(file_info['columns'])
            first_file = False
        else:
            common_columns = common_columns.intersection(set(file_info['columns']))
    
    print(f"\nFound {len(common_columns)} common columns across all files")
    print("Common columns:", sorted(common_columns))
    
    return common_columns

# Run the analysis
if __name__ == "__main__":
    main_data_dir = "C:\Project\Dataset"  # Your main dataset folder
    
    print("Starting analysis of CIC-DDoS2019 dataset structure...")
    analysis = analyze_dataset_structure(main_data_dir)
    common_cols = find_common_columns(analysis)
    
    # Save the results to a file for reference
    with open("../data/column_analysis_report.txt", "w") as f:
        f.write("CIC-DDoS2019 Column Analysis Report\n")
        f.write("=" * 50 + "\n")
        for filename, info in analysis.items():
            f.write(f"\nFile: {filename}\n")
            f.write(f"Columns: {info['column_count']}\n")
            f.write(f"Label present: {info['label_column'] is not None}\n")
            f.write(f"Columns: {info['columns']}\n")