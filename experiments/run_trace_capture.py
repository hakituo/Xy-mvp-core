import subprocess
import time
import os
import sys
import glob
import json
import re

def capture_data():
    print("=== Starting Trace & SMI Capture (Real Workload) ===")
    
    # 1. Start Experiment (Async)
    # Run Exp 3 with 1 request to get a clean single trace
    cmd = [sys.executable, "comprehensive_experiment.py", "--workload", "real", "--exp", "3", "--n_requests", "1"]
    
    # We need to run this in the mvp_core/experiments directory
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=cwd, bufsize=1, universal_newlines=True)
    
    import threading
    def stream_reader(pipe, label, is_stderr=False):
        try:
            for line in iter(pipe.readline, ''):
                try:
                    print(f"[{label}] {line.strip()}")
                except UnicodeEncodeError:
                    # Fallback for Windows consoles that can't handle some chars
                    clean_line = line.strip().encode('ascii', 'ignore').decode('ascii')
                    print(f"[{label}] {clean_line}")
                if is_stderr:
                    # Capture stderr for error checking
                    pass 
        except Exception:
            pass

    # Start threads to stream output
    t_out = threading.Thread(target=stream_reader, args=(process.stdout, "EXP_OUT"))
    t_out.daemon = True
    t_out.start()
    
    t_err = threading.Thread(target=stream_reader, args=(process.stderr, "EXP_ERR", True))
    t_err.daemon = True
    t_err.start()

    print("Experiment launched. Waiting for model loading (approx 10s)...")
    
    # Monitor for SMI capture
    # We'll try to capture when we see "Exp 3" in stdout or just after a delay
    smi_captured = False
    
    start_time = time.time()
    while process.poll() is None:
        # Check output (non-blocking read is hard without threads, so we just sleep and snapshot)
        time.sleep(2)
        elapsed = time.time() - start_time
        
        # Capture SMI around 15s mark (assuming models loaded and task running)
        if elapsed > 15 and not smi_captured:
            print("Attempting SMI capture...")
            try:
                smi_out = subprocess.check_output("nvidia-smi", shell=True, text=True)
                with open("nvidia_smi_snapshot.txt", "w") as f:
                    f.write(smi_out)
                print(">>> NVIDIA-SMI Snapshot Captured!")
                smi_captured = True
            except Exception as e:
                print(f"SMI Capture failed: {e}")
        
        if elapsed > 120: # Timeout
            print("Timeout! Killing process.")
            process.kill()
            break
            
    process.wait()
    print("Experiment Process Finished.")
    
    if process.returncode != 0:
        print("Experiment Error (check logs above).")
    else:
        print("Experiment Success.")
        
    # 2. Find and Parse Log
    # Log is in ../../logs relative to experiments/
    log_dir = os.path.join(cwd, "..", "..", "logs")
    log_files = glob.glob(os.path.join(log_dir, "benchmark_trace_*.log"))
    
    trace_events = []
    
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"Reading log: {latest_log}")
        
        with open(latest_log, 'r', encoding='utf-8') as f:
            for line in f:
                # We look for lines related to our task
                # Since logging is text, we need to regex parse relevant lines
                # Format: [Time] [Level] [TraceID] [Logger] Message
                # We want to reconstruct: start, complete events
                
                # Sample: [2025-12-08 06:00:00] [INFO] [TraceID:task_...] [ExperimentRunner] Starting task...
                # We map these to JSON events
                
                # Updated mapping based on actual logs
                if "executing task" in line:
                    ts = extract_ts(line)
                    # Extract task ID
                    # ... Worker worker-0 executing task task_97304782_0
                    task_id = line.split("executing task")[-1].strip()
                    trace_events.append({"ts": ts, "event": "task_started", "task_id": task_id, "details": line.strip()})
                elif "Task" in line and "completed" in line:
                    ts = extract_ts(line)
                    # ... Task task_97304782_0 completed
                    task_id = line.split("Task")[-1].split("completed")[0].strip()
                    trace_events.append({"ts": ts, "event": "task_completed", "task_id": task_id, "details": line.strip()})
                elif "scheduled - ID:" in line:
                     ts = extract_ts(line)
                     # ... Task scheduled - ID: task_97304782_0, Name: llm_-1
                     task_id = line.split("ID:")[-1].split(",")[0].strip()
                     trace_events.append({"ts": ts, "event": "task_scheduled", "task_id": task_id, "details": line.strip()})
                     
        # If we found events, dump them
        if trace_events:
            with open("captured_trace.json", "w") as f:
                json.dump(trace_events, f, indent=2)
            print(f"Generated {len(trace_events)} trace events.")
    else:
        print("No log file found.")

def extract_ts(line):
    # Extract timestamp from [2025-...]
    try:
        ts_str = line.split(']')[0].strip('[')
        dt = time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
        return dt
    except:
        return time.time()

if __name__ == "__main__":
    capture_data()
