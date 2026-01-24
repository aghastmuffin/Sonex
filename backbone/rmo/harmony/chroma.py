import librosa
from backbone.data.frames import ChromaFrame

def extract_chroma(cqt_frame):
    chroma = librosa.feature.chroma_cqt(
        C=cqt_frame.magnitudes,
        sr=cqt_frame.sr,
        hop_length=cqt_frame.hop_length
    )
    return ChromaFrame(chroma, cqt_frame.sr, cqt_frame.hop_length)
