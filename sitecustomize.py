import torch
import torch._ops

def _silu_and_mul(out, x):
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])

def _rms_norm(out, x, weight, epsilon):
    var = x.pow(2).mean(-1, keepdim=True)
    out.copy_(x * torch.rsqrt(var + epsilon) * weight)

def _fused_add_rms_norm(x, residual, weight, epsilon):
    x.add_(residual)
    var = x.pow(2).mean(-1, keepdim=True)
    x.copy_(x * torch.rsqrt(var + epsilon) * weight)

class DummyOp:
    def __init__(self, fn):
        self.fn = fn
        self.default = fn
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

_FALLBACK_OPS = {
    "init_cpu_memory_env": DummyOp(lambda *a, **k: None),
    "silu_and_mul": DummyOp(_silu_and_mul),
    "rms_norm": DummyOp(_rms_norm),
    "fused_add_rms_norm": DummyOp(_fused_add_rms_norm),
}

orig_getattr = torch._ops._OpNamespace.__getattr__

def safe_getattr(self, name):
    ns_name = getattr(self, "_name", getattr(self, "name", None))
    if ns_name in ("_C", "vllm", None) and name in _FALLBACK_OPS:
        return _FALLBACK_OPS[name]
    return orig_getattr(self, name)

torch._ops._OpNamespace.__getattr__ = safe_getattr


