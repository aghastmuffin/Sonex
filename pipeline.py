from backbone.rmo.io import load_audio
from backbone.rmo.spectral.cqt import compute_cqt
from backbone.rmo.harmony.chroma import extract_chroma

def analyze_file(path):
    audio = load_audio(path)
    cqt = compute_cqt(audio)
    chroma = extract_chroma(cqt)
