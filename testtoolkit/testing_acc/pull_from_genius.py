import requests
from bs4 import BeautifulSoup

def get_lyrics(url):
    # Set a User-Agent to avoid being blocked as a bot
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return "Error: Could not reach the page."

    soup = BeautifulSoup(response.text, 'html.parser')

    # Genius stores lyrics in containers with 'Lyrics__Container' in the class name
    lyric_containers = soup.find_all('div', class_=lambda x: x and 'Lyrics__Container' in x)

    if not lyric_containers:
        return "Lyrics not found. The page structure might have changed."

    lyrics = ""
    for container in lyric_containers:
        # .get_text(separator="\n") preserves the line breaks
        lyrics += container.get_text(separator="\n")

    return lyrics.strip()

# Example Usage
songs = {"22": "https://genius.com/Taylor-swift-22-lyrics", "alambre_pua_bb": "https://genius.com/Bad-bunny-alambre-pua-lyrics", "alambre_pua-en": "https://genius.com/Genius-english-translations-bad-bunny-alambre-pua-english-translation-lyrics", "handsup": "https://genius.com/Blood-orange-hands-up-lyrics", "kids": "https://genius.com/Mgmt-kids-lyrics", "bugatti": "https://genius.com/Anuel-aa-bugatti-lyrics", "lastnight": "https://genius.com/Morgan-wallen-last-night-lyrics", "entolao": "https://genius.com/Fino-como-el-haze-and-jhayco-en-to-lao-lyrics", "make it to the morning": "https://genius.com/Partynextdoor-m-a-k-e-i-t-t-o-t-h-e-m-o-r-n-i-n-g-lyrics", "need you now": "https://genius.com/Lady-a-need-you-now-lyrics", "penny lane": "https://genius.com/The-beatles-penny-lane-lyrics", "presiento": "https://genius.com/Rkm-and-ken-y-presiento-lyrics", "Prohabition(ley-seca)-en": "https://genius.com/Genius-english-translations-jhay-cortez-and-anuel-aa-ley-seca-english-translation-lyrics", "Ley Seca": "https://genius.com/Jhayco-and-anuel-aa-ley-seca-lyrics", "experimento": "https://genius.com/Myke-towers-experimento-lyrics", "thailand": "https://genius.com/Malcolm-todd-thailand-lyrics", "tuchat": "https://genius.com/Quevedo-tuchat-lyrics", "un sueno": "https://genius.com/Alex-gargolas-and-rkm-and-ken-y-un-sueno-lyrics", "lo siento bb": "https://genius.com/Tainy-bad-bunny-and-julieta-venegas-lo-siento-bb-lyrics", "oa": "https://genius.com/Anuel-aa-quevedo-and-maluma-oa-lyrics", "zip up my fly": "https://genius.com/Malcolm-todd-zip-up-my-fly-lyrics"}
for songdata in songs: 
    with open(f"lyrics_genius/{songdata}.txt", "w") as f:
        f.write(get_lyrics(songs[songdata]))
