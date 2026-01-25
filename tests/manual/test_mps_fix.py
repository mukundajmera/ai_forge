import sys
import os

# Import the module that should have the fix
try:
    print("Importing training.forge...")
    from training import forge
    print("Successfully imported training.forge")
except ImportError as e:
    print(f"Failed to import training.forge: {e}")
    # Try adding current dir to path
    sys.path.append(os.getcwd())
    try:
        from training import forge
        print("Successfully imported training.forge after sys.path update")
    except ImportError as e2:
        print(f"Still failed to import: {e2}")
        sys.exit(1)

import torch

def test_qr_mps_with_fix():
    print(f"Checking PYTORCH_ENABLE_MPS_FALLBACK env var: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    
    if not torch.backends.mps.is_available():
        print("MPS not available. Skipping.")
        sys.exit(0)

    device = torch.device("mps")
    A = torch.randn(50, 50, device=device)
    
    try:
        print("Attempting torch.linalg.qr on MPS (expecting success via fallback)...")
        Q, R = torch.linalg.qr(A)
        print("PASS: torch.linalg.qr executed successfully with fix.")
    except RuntimeError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_qr_mps_with_fix()
