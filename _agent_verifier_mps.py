import torch
import os
import sys

def test_qr_mps():
    print(f"PyTorch Version: {torch.__version__}")
    if not torch.backends.mps.is_available():
        print("MPS not available. This test requires Apple Silicon.")
        # If we can't test it, we can't verify failure/success locally if not on Mac.
        # But the User Info says "The USER's OS version is mac."
        # However, checking if the container has access to MPS is crucial.
        sys.exit(0) 

    device = torch.device("mps")
    print(f"Testing on device: {device}")
    
    # Create a random matrix
    A = torch.randn(50, 50, device=device)
    
    try:
        print("Attempting torch.linalg.qr on MPS...")
        Q, R = torch.linalg.qr(A)
        print("PASS: torch.linalg.qr executed successfully on MPS.")
        sys.exit(0)
    except RuntimeError as e:
        print(f"CAUGHT ERROR: {e}")
        if "aten::linalg_qr.out" in str(e) and "not currently implemented for the MPS device" in str(e):
            print("FAIL: Reproduced expected MPS error.")
            sys.exit(1)
        else:
            print(f"FAIL: Unexpected error: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"FAIL: Unexpected exception: {type(e)} - {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check if fallback is set
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK = {fallback}")
    test_qr_mps()
