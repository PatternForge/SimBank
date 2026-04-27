import subprocess
import sys


def run(state):
    subprocess.run([sys.executable, "-m", "SimBank"], check=True)
    state.mark("python_sot", "success")

