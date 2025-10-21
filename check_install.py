print("=== VANET DDoS Project - Installation Check ===\n")

# Check Python version
import sys
print(f"Python Version: {sys.version}\n")

# Check core data science packages
try:
    import numpy
    print(f"✓ NumPy: {numpy.__version__}")
except ImportError:
    print("✗ NumPy: NOT INSTALLED")

try:
    import pandas
    print(f"✓ Pandas: {pandas.__version__}")
except ImportError:
    print("✗ Pandas: NOT INSTALLED")

try:
    import sklearn
    print(f"✓ Scikit-learn: {sklearn.__version__}")
except ImportError:
    print("✗ Scikit-learn: NOT INSTALLED")

# Check visualization packages
try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError:
    print("✗ Matplotlib: NOT INSTALLED")

try:
    import seaborn
    print(f"✓ Seaborn: {seaborn.__version__}")
except ImportError:
    print("✗ Seaborn: NOT INSTALLED")

# Check PyTorch and GPU access
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  CUDA: Not available (using CPU)")
except ImportError:
    print("✗ PyTorch: NOT INSTALLED")

print("\n=== Check Complete ===")