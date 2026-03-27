import transformers
import torch
import datasets
import sklearn
import streamlit
import matplotlib

print(f"✓ transformers : {transformers.__version__}")
print(f"✓ torch        : {torch.__version__}")
print(f"✓ datasets     : {datasets.__version__}")
print(f"✓ sklearn      : {sklearn.__version__}")
print(f"✓ streamlit    : {streamlit.__version__}")
print(f"✓ matplotlib   : {matplotlib.__version__}")
print(f"")
print(f"✓ CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  Running on CPU — training will take ~45 min/epoch (totally fine)")

print("\n✅ All imports successful. Environment is ready.")