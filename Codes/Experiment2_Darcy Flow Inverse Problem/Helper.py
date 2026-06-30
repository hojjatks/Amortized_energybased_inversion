import os, psutil, torch

def print_memory(tag: str = ""):
    """
    Prints cumulative memory usage:
      - CPU RSS from psutil (MB)
      - For each CUDA GPU:
          - PyTorch allocated & reserved memory (MB)
          - Driver-level used/total memory (MB)
    """
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"[{tag}] CPU RSS: {rss_mb:.2f} MB")

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        for d in range(num_devices):
            allocated_mb = torch.cuda.memory_allocated(d) / 1024**2
            reserved_mb  = torch.cuda.memory_reserved(d)  / 1024**2

            free_b, total_b = torch.cuda.mem_get_info(d)
            free_mb, total_mb = free_b/1024**2, total_b/1024**2
            used_driver_mb = total_mb - free_mb

            print(f"    GPU {d}: alloc={allocated_mb:.2f} MB, "
                  f"reserved={reserved_mb:.2f} MB | "
                  f"driver used/total={used_driver_mb:.2f}/{total_mb:.2f} MB")
    else:
        print("    No CUDA devices available.")
