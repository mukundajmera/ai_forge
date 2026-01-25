import gguf
from enum import IntEnum

print("Testing monkeypatch...")

try:
    print(f"AFMOE before: {getattr(gguf.MODEL_ARCH, 'AFMOE', 'Missing')}")
except Exception as e:
    print(f"Access error: {e}")

# Attempt to patch
try:
    # IntEnum members are usually set.
    # We can try extending it from scratch or just setting attributes?
    # gguf.MODEL_ARCH is a class.
    
    # Strategy 1: setattr
    # gguf.MODEL_ARCH.AFMOE = 9999
    # This might fail if it's a frozen enum.
    pass
except Exception as e:
    print(f"Setattr failed: {e}")

# Strategy 2: Replace the class with a permissive one
original_arch = gguf.MODEL_ARCH
class PatchedArch:
    def __getattr__(self, name):
        if hasattr(original_arch, name):
            return getattr(original_arch, name)
        return 9999 # Dummy value

# Check if we can overwrite gguf.MODEL_ARCH
# gguf.MODEL_ARCH = PatchedArch() 
# But the script imports gguf.
# And uses gguf.MODEL_ARCH.AFMOE (enum member).
# Enum members are instances of the Enum class.

# If we just need the value to be readable as gguf.MODEL_ARCH.AFMOE
# And it should be an int or comparable.

# Let's see if we can just define the missing ones.
missing = ["AFMOE", "MISTRAL3", "DREAM", "LLAMA4"]
start_val = 1000

for m in missing:
    try:
        # extend_enum is a trick often used
        import aenum
        # But we don't have aenum.
        pass
    except:
        pass

# Fallback: Just define attributes on the class?
try:
    setattr(gguf.MODEL_ARCH, 'AFMOE', 9999)
    print("setattr worked!")
    print(f"AFMOE: {gguf.MODEL_ARCH.AFMOE}")
except TypeError as e:
    print(f"setattr failed: {e}") 
    # Enums are allowed to have attributes if they are not members? 
    # But usually access by dot returns member.

