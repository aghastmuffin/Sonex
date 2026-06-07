import subprocess
from tkinter.ttk import *
import tkinter as tk
root = tk.Tk()

def install():
    #check brew status
    try:
        proc = subprocess.check_output("brew", stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        if "command not found" in e.output.decode("utf-8"):
            subprocess.call("", shell=True)

    #check pyver/pyver install via pyenv

    #install deps avaliable via pip
    subprocess.call(["pip", "install", "-r", "requirements.txt"])
#install pyenv
#install 3.9
#

subprocess.popen("deactivate") #only works for pip envs anyawys

#conda create -n mfa-env python=3.10 -y
#conda activate mfa-env
#conda install -c conda-forge montreal-forced-aligner -y

root.mainloop()