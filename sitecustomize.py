import torch
import torch._ops

# Handle missing CPU extension attributes gracefully across all processes
orig_getattr = torch._ops._OpNamespace.__getattr__
def safe_getattr(self, name):
    if name == "init_cpu_memory_env":
        return lambda *args, **kwargs: None
    return orig_getattr(self, name)

torch._ops._OpNamespace.__getattr__ = safe_getattr
