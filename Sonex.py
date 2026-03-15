import subprocess
import os
import pathlib as pl
#Orchestrator

outp = subprocess.run(["python", "ui/gui.py"], check=True, text=True, capture_output=True)
if "Done" in outp.stdout:
    subprocess.run("python", "ui/frase.py", check=True, text=True, capture_output=True)
else:
    print("generator didn't run successfully :(")
