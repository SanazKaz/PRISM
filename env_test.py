import torch

print(f"Testing Environment: PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Size just over INT_MAX (2^31)
# 2,147,483,648 is the limit; we'll use 2.2 billion.
size = 2_200_000_000 

try:
    print(f"Allocating dummy tensor of size {size} (~2.2GB VRAM)...")
    # We use a boolean mask to mimic the 'adj' matrix in your dynamics.py
    mask = torch.zeros(size, dtype=torch.bool, device='cuda')
    mask[0] = True
    mask[-1] = True
    
    print("Running torch.where (The INT_MAX hurdle)...")
    indices = torch.where(mask)
    
    print(f"✅ TEST PASSED: Found indices at {indices[0].tolist()}")
    print("64-bit indexing is working. Large pockets will not crash this environment.")

except RuntimeError as e:
    if "INT_MAX" in str(e) or "nonzero" in str(e):
        print(f"❌ TEST FAILED: Still hitting the INT_MAX limit: {e}")
    else:
        print(f"❌ CUDA ERROR: {e}")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")