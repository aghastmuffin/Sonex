import os, re
delportion = False
for filename in os.listdir("lyrics_study/lyrics"):
    with open ("lyrics_study/lyrics/" + filename, "r") as f:
        lines = f.read()
    with open ("lyrics_study/clean_lyrics/" + filename, "wb") as f:
        lines = lines.replace('\n', ' ')
        first_bracket = lines.find('[')
        if first_bracket != -1:
            lines = lines[first_bracket:]
        f.write(re.sub(r'\[[^\]]*\]', '', lines))