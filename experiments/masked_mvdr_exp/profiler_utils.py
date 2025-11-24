import time
import tracemalloc
import json
import os
import contextlib

@contextlib.contextmanager
def profile_performance(stats_path):
    """
    Context manager to measure Wall Time and Peak RAM.
    Writes a JSON file to stats_path upon exit.
    
    Usage:
        with profile_performance("output_stats.json"):
            run_heavy_computation()
    """
    # 1. Start Timers
    tracemalloc.start()
    start_time = time.perf_counter()
    
    try:
        yield
    finally:
        # 2. Stop Timers
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # 3. Calculate Stats
        duration = end_time - start_time
        peak_mb = peak / (1024 * 1024) # Convert Bytes to MB
        
        stats = {
            "duration_sec": round(duration, 4),
            "peak_ram_mb": round(peak_mb, 2)
        }
        
        # 4. Save to JSON
        # Ensure directory exists before writing
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"   [Profile] Time: {duration:.2f}s | Peak RAM: {peak_mb:.2f}MB")