#!/usr/bin/env python3
"""Launch the optimization as a fully detached process on Windows."""
import subprocess
from pathlib import Path

project_root = Path(__file__).parent.parent
log_path = project_root / "data" / "results" / "optimization_run.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

python_exe = r"D:\Program Files\Python\python.exe"
script = str(project_root / "scripts" / "run_full_optimization.py")

# Use CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS so it survives
creation_flags = (
    subprocess.CREATE_NEW_PROCESS_GROUP
    | subprocess.DETACHED_PROCESS
    | subprocess.CREATE_NO_WINDOW
)

proc = subprocess.Popen(
    [python_exe, "-u", script, "--resume", "--checkpoint-every", "500"],
    cwd=str(project_root),
    stdout=open(log_path, "w", encoding="utf-8"),
    stderr=subprocess.STDOUT,
    creationflags=creation_flags,
)

print(f"Optimization launched as PID {proc.pid}")
print(f"Log file: {log_path}")
print(f"Use: tail -f {log_path}  to monitor progress")
