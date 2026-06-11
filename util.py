def cuda_cleanup(device=None):
    import gc
    import torch

    # 先断开全局 scheduler 引用（避免 optimizer/param 仍被持有）
    for _n in ["scheduler_d", "scheduler_g"]:
        if _n in globals():
            try:
                globals()[_n] = None
                del globals()[_n]
            except Exception:
                globals()[_n] = None

    # 多轮 GC，尽量清循环引用
    for _ in range(3):
        gc.collect()

    if torch.cuda.is_available():
        try:
            if device is None:
                device = torch.cuda.current_device()
        except Exception:
            device = 0

        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass

        # 清缓存 + IPC 回收
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        # 重置统计（不释放显存，但便于你看“是否还在涨”）
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        try:
            torch.cuda.reset_accumulated_memory_stats(device)
        except Exception:
            pass

    # 再来一轮 GC
    for _ in range(2):
        gc.collect()

def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)
