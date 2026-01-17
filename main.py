import multiprocessing
import time
import os
import numpy as np
import hashlib

import PyQt6
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtWidgets import QApplication, QMessageBox, QWidget
import sys

from backbone.rmo.MEL import save_mel_spectrogram, render_mel, render_mel_frames
from backbone.ltra.letra_toolkit import separate
from audio.audiocontext import AudioContext
#Config

FILE = "ADV.mp3" 

#Functions

def sysalert(title:str, message:str, parent=None, severity:int=0):
    if severity == 0:
        QMessageBox.information(
            parent,                   # Parent widget (None makes it a top-level window)
            title,   # Dialog title
            message # Dialog message
        )
    elif severity == 1:
        QMessageBox.warning(
            parent,                   # Parent widget (None makes it a top-level window)
            title,   # Dialog title
            message # Dialog message
        )
    elif severity == 2:
        QMessageBox.critical(
            parent,                   # Parent widget (None makes it a top-level window)
            title,   # Dialog title
            message # Dialog message
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Example usage
    try:
        HASH = hashlib.md5(open(FILE,'rb').read()).hexdigest()[:10]
        os.makedirs(f"output_{HASH}", exist_ok=False)
        if os.path.isfile(FILE):
            os.rename(FILE, os.path.join(f"output_{HASH}", FILE))
        else:
            sysalert("File Not Found", f"Audio file {FILE} not found in current directory.", severity=0)
            
        os.chdir(f"output_{HASH}")

        save_mel_spectrogram(FILE)
    except FileExistsError:
        sysalert("Directory Exists", f"Output directory output_{HASH} already exists. Please remove or choose a different file.", severity=2)
    except FileNotFoundError:
        sysalert("File Not Found", f"Audio file {FILE} could not found in current directory. This error occured during initialization of program, not in finding the file within a subfolder or directory.", severity=2)

