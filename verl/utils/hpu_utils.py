import os
import torch

try:
    import habana_frameworks.torch.core as htcore
except:
    pass

def is_real_cuda_device_available():
    return hasattr(torch._C, "_cuda_getDeviceCount")

def is_lazy_mode():
    lazy_mode = int(os.getenv("PT_HPU_LAZY_MODE", "0"))
    assert 0 <= lazy_mode <= 2, f"PT_HPU_LAZY_MODE is set incorrectly"
    return lazy_mode in (1, 2)

def is_gpu_migration_tool_enabled():
    gpu_migration_tool = int(os.getenv("PT_HPU_GPU_MIGRATION", "0"))
    assert 0 <= gpu_migration_tool <= 1, f"PT_HPU_GPU_MIGRATION is set incorrectly"
    return gpu_migration_tool == 1

def call_mark_step():
    if is_real_cuda_device_available():
        return
    if is_lazy_mode():
        htcore.mark_step()
    # else:
    #     torch._dynamo.graph_break()

def hpu_empty_cache():
    pass

torch.hpu.empty_cache = hpu_empty_cache
