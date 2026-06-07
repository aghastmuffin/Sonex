# Sonex - Escucha el Futuro
Por: Levi Brown, representado como "levitk", de lado de taeson.co.

# Que?
Sonex, principalmente es un bien reproductor de música, pero hay charactarísticas que establecer aparte de otrás. Sonex tiene un dinámico sistema de renderizado pa' reconociendo los palabaras, notas, y ritmo. Sonex es un experimento, y es compuesto por artículos de investigación, algunos de los cuales se enumeran a continuación

# artículos
Lyu, K.-M., Lyu, R. and Chang, H.-T. (2024). Real-time multilingual speech recognition and speaker diarization system based on Whisper segmentation (1) [Computer Software]. PeerJ Computer Science, [online] 10, p.e1973. doi:[doi.org/10.7717/peerj-cs.1973](https://doi.org/10.7717/peerj-cs.1973.).

Bernard, M. and Titeux, H. (2021). Phonemizer: Text to Phones Transcription for Multiple Languages in Python (phoneizer) [Computer Software]. Journal of Open Source Software, 6(68), p.3958. doi:[doi.org/10.21105/joss.03958](https://doi.org/10.21105/joss.03958).

# Como Se Funciona?

## MFA: timestamps por fonema

Sonex ahora soporta salida explícita por fonema desde MFA con dos modos:

1. Alinear audio + transcript y generar JSONs de palabras/fonemas.
2. Extraer fonemas directamente desde un TextGrid ya generado.

### Comando 1: Alineación completa + salida de fonemas

Desde la raíz del repo:

python -m backbone.ltra._mfa_aligner align deptest/ojitos_lindos --acoustic spanish_mfa --dictionary spanish_mfa --allow-fuzzy

Archivos de salida esperados dentro de la carpeta del track:

- mfa_vocals_whisper_segments.json (palabras con start/end)
- mfa_vocals_phone_segments.json (palabras con phones y phone_segments para highlighting sub-palabra)
- mfa_vocals_phone_timestamps.json (lista plana de fonemas con start/end)

### Comando 2: Extraer fonemas desde TextGrid existente

python -m backbone.ltra._mfa_aligner export-phones deptest/ojitos_lindos/_mfa_out/vocals_000.TextGrid --out deptest/ojitos_lindos/mfa_vocals_phone_timestamps.json

Opcional: incluir silencios (sp/sil):

python -m backbone.ltra._mfa_aligner export-phones deptest/ojitos_lindos/_mfa_out/vocals_000.TextGrid --include-silence

### Comando 3: Completar phones y crear phone_segments en JSON existente

python -m backbone.ltra._mfa_aligner segment-phones deptest/ojitos_lindos/mfa_vocals_phone_segments.json --phonemizer-lang es


# Sobre DataStore
Operado por Taeson.co, se usa las transcriptiones que generarás y otra informacion sobre la cancion a determinar la exactitud de su audio al proveedores como Genius, LyricsMatch, LyricsTranslate, etc.