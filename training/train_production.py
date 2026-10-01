"""Forwarding entrypoint for root workspace execution."""
import os
import sys
import subprocess

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_script = os.path.join(root_dir, "FinPulse", "training", "train_production.py")
    res = subprocess.run([sys.executable, target_script] + sys.argv[1:], cwd=os.path.join(root_dir, "FinPulse"))
    sys.exit(res.returncode)
