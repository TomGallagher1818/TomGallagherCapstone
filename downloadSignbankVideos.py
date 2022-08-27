# Import Required Module
from lib2to3.pytree import Base
import requests
from bs4 import BeautifulSoup
import re
from os.path import exists
from string import ascii_uppercase
 
# Web URL
WORD_TO_URL = {}
Base_url = "https://auslan.org.au"

for letter in ascii_uppercase:
    Web_url = "https://auslan.org.au/dictionary/search/?query=" + letter + "&category=all"
    # Get URL Content
    r = requests.get(Web_url)
    
    # Parse HTML Code
    soup = BeautifulSoup(r.content, 'html.parser')
    
    # List of all video tag
    video_tags = soup.findAll('a')

    if len(video_tags) != 0:
        for video_tag in video_tags:
            videoTagString = str(video_tag)
            append_url = re.search(r'/dictionary.+.html', videoTagString)
            if(append_url):
                append_url = append_url.group()
                word = append_url.split("/", -1)[-1].split("-")[0]
                url = Base_url + append_url
                WORD_TO_URL[word] = url
                

    for word in WORD_TO_URL.keys():
        print("WORD: " + word)
        url = WORD_TO_URL[word]
        video_path = "SignBankVideos/" + word + ".mp4"
        if not exists(video_path):
            response = requests.get(url)
            if response.status_code == 200:
                html = BeautifulSoup(response.content, 'html.parser')
                video_tags = html.findAll('video')
                if len(video_tags) != 0:
                    for video_tag in video_tags:
                        source = str(video_tag)
                        url = re.search(r'https://media.auslan.org.au/.+.mp4',source)
                        url = str(url.group())
                        url = url.split(".mp4",1)[0] + ".mp4"
                        video_url = requests.get(url, stream = True)

                        if not exists(video_path):
                        #download
                            with open(video_path, 'wb') as f:
                                for chunk in video_url.iter_content(chunk_size=320*240):
                                    if chunk:
                                        f.write(chunk)
                            print( word + " downloaded!\n")
        else:
            print(word + " already downloaded!\n")