from backbone.ltra.argos_tranlsate import translate_file

out = translate_file("anuel_bugatti/vocals_whisper_segments.json", from_lang="es", to_lang="en", verbose=True)
print(out)