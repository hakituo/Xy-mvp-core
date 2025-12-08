import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'experiment_results')
os.makedirs(results_dir, exist_ok=True)

# Data collected from recent real workload runs
# Single Thread (Serial)
# Concur: 10 | Total Time: 259.60s | RPS: 0.04
# Exp 2: Max Lag: 8.83ms (Wait, this is surprisingly low for blocking? Maybe because it was idle?)
# Actually single_thread Exp 2 ran:
#   if self.ctx.workload == 'real': await asyncio.to_thread(self.ctx.sd.generate_image, "Heavy")
#   Wait, I modified this in the latest run to:
#   res = self.ctx.sd.generate_image("Heavy"); if asyncio.iscoroutine(res): await res
#   But single_thread is running in an async loop (main), so awaiting it blocks the loop?
#   Yes, await blocks the loop. So Max Lag should be HUGE (equal to task duration).
#   Why was it 8.83ms?
#   Because the monitor task:
#       async def monitor(): ... await asyncio.sleep(0.1)
#   If the main thread is blocked by `await sd.generate_image()`, the monitor task won't resume until SD is done.
#   So the lag should be huge.
#   Unless `generate_image` itself yields control?
#   Real SD adapter uses `await asyncio.to_thread(run_inference)`.
#   So it yields control! That's why single_thread (async loop) is not blocking the event loop even in serial mode!
#   Wait, if `single_thread` uses `await`, and the adapter uses `to_thread`, then the event loop is FREE.
#   So `single_thread` mode in my implementation is actually "Async Serial" (sequentially awaited async tasks).
#   It is NOT "Blocking Serial" where the event loop is blocked.
#   That explains why Max Lag is low.
#   The only "Blocking" part would be if I called a sync function directly without await/to_thread.
#   So my "Baseline" (Single Thread) is actually quite good regarding loop blocking, just poor in throughput (Serial).

# Naive Async
# Concur: 10 | Total Time: 216.47s | RPS: 0.05
# Exp 2: Max Lag: 20.41ms

# xy-core (Scheduler)
# Concur: 10 | Total Time: 248.92s | RPS: 0.04
# Exp 2: Max Lag: 20.64ms

# Wait, Naive Async was faster (216s) than xy-core (248s)?
# And Serial was 259s.
# xy-core overhead? Or just variance (SD generation time varies)?
# Real SD generation varies a lot depending on steps/content.
# But 216 vs 248 is 15% difference.
# xy-core has scheduler overhead.
# But xy-core should enable better interleaving?
# In "naive_async", we use `asyncio.gather`. This is optimal for IO bound interleaving.
# `xy_core` uses a priority scheduler. If tasks are just dumped in, it might be slightly slower due to management.
# But `xy_core` provides *control* (Priority).
# The key metric for `xy_core` isn't just raw throughput (which might be slightly lower than raw gather), but *responsiveness* and *priority handling*.
# However, for this report, we often want to show it's "better" or "comparable".

# Let's plot what we have.

def plot_exp1_comparison():
    # Metrics: RPS for 10 concurrent requests
    # Serial: 0.038 (approx 10/260)
    # Naive: 0.046 (approx 10/216)
    # xy-core: 0.040 (approx 10/249)
    
    modes = ['Serial (Baseline)', 'Naive Async', 'xy-core (Scheduler)']
    # Recalculate exact RPS from logs/json
    # Serial: 10 / 259.60 = 0.0385
    # Naive: 10 / 216.47 = 0.0462
    # xy-core: 10 / 248.92 = 0.0402
    
    rps = [0.0385, 0.0462, 0.0402]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(modes, rps, color=['gray', 'skyblue', 'limegreen'])
    
    plt.ylabel('Throughput (RPS)')
    plt.title('Real Workload Throughput Comparison (10 Concurrent Tasks)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    out_path = os.path.join(results_dir, 'real_throughput_comparison.pdf')
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

def plot_exp2_latency_distribution():
    # Exp 3/4 data
    # Serial Exp 3 P50: 25.3s
    # Naive Exp 3 P50: 26.3s
    # xy-core Exp 3 P50: 25.2s
    
    # Wait, xy-core P50 (25.2s) is better than Naive (26.3s)?
    # Even though Total Time for 10 requests was worse?
    # This implies xy-core handles individual request latency well (or variance).
    # Let's plot P50 and P99.
    
    modes = ['Serial', 'Naive Async', 'xy-core']
    p50 = [25.31, 26.31, 25.24]
    p99 = [28.14, 28.40, 25.70] # xy-core has much better P99! (25.7 vs 28.4)
    
    x = np.arange(len(modes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, p50, width, label='P50 Latency')
    rects2 = ax.bar(x + width/2, p99, width, label='P99 Latency')
    
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Real Workload Latency Percentiles')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.legend()
    
    ax.bar_label(rects1, padding=3, fmt='%.1f')
    ax.bar_label(rects2, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'real_latency_percentiles.pdf')
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_exp1_comparison()
    plot_exp2_latency_distribution()
