import os
import re

path = 'wgp.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add helper function
helper_code = """
def safe_cuda_cleanup():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.synchronize()
        except Exception: pass
        try: torch.cuda.empty_cache()
        except Exception: pass
        try: torch.cuda.ipc_collect()
        except Exception: pass
"""

if "def safe_cuda_cleanup():" not in content:
    import_marker = "import torch"
    pos = content.find(import_marker)
    if pos != -1:
        line_end = content.find("\n", pos)
        content = content[:line_end+1] + helper_code + content[line_end+1:]

# 2. Replace naked cleanup calls with safe_cuda_cleanup()
# We want to replace blocks like:
# if torch.cuda.is_available():
#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()

# Let's search for sequences.
# First, replace individual ones.

# Replace torch.cuda.empty_cache()
content = content.replace("torch.cuda.empty_cache()", "safe_cuda_cleanup()")

# Clean up redundant check_available blocks if they only contained one call
# e.g. "if torch.cuda.is_available():\n    safe_cuda_cleanup()"
content = re.sub(r"if torch\.cuda\.is_available\(\):\s+safe_cuda_cleanup\(\)", "safe_cuda_cleanup()", content)

# Clean up previous turn's try-except blocks
content = re.sub(r"try: safe_cuda_cleanup\(\)\s+except Exception: pass", "safe_cuda_cleanup()", content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("wgp.py updated with safe_cuda_cleanup helper v2")
