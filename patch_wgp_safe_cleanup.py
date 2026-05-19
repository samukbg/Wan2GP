import os

path = 'wgp.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Helper function to be added at the top
helper_code = """
def safe_cuda_cleanup():
    if torch.cuda.is_available():
        try: torch.cuda.synchronize()
        except Exception: pass
        try: torch.cuda.empty_cache()
        except Exception: pass
        try: torch.cuda.ipc_collect()
        except Exception: pass
"""

# Find a good spot for the helper (after imports)
import_end = content.find("from shared.utils.resource_guard")
if import_end != -1:
    line_end = content.find("\n", import_end)
    content = content[:line_end+1] + helper_code + content[line_end+1:]

# Replace various patterns of CUDA cleanup with the safe helper
import re

# Replace torch.cuda.empty_cache()
content = re.sub(r"torch\.cuda\.empty_cache\(\)", "safe_cuda_cleanup()", content)

# Note: Some might already be in try/except blocks I added earlier.
# re.sub will replace them, which is fine, safe_cuda_cleanup is even safer.
# However, I should be careful not to create "try: safe_cuda_cleanup()" which is redundant but okay.

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("wgp.py updated with safe_cuda_cleanup helper")
