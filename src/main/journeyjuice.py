# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
from pathlib import Path

try:
    import tensorflow 
except ImportError:
    pass

# import chatGPT
import subprocess

# import recording modules
import sounddevice as sd
from scipy.io.wavfile import write
from IPython.display import Audio, display

# Google Event API
from serpapi import GoogleSearch

# Text2speech related modules 
from gtts import gTTS
import os
from IPython.display import Audio
from pydub import AudioSegment
from pydub.playback import play

# Map
import googlemaps
import folium

# Keys:
googlemap_key = 'AIzaSyCqONC34oYyMeq8qhEGp-BCWYFHcpNkLRs' #Anant
#ChatGPT - Dora - masked
googleevents_key = "9ec617496aef876ebae248c1ff5ee771ccf9dd4b0762d1524d93f35557196949" # Dora


################### Record audio 
def record():
    duration = 5  # seconds
    filename = 'recording.wav'
    print(f"Recording audio for {duration} seconds...")
    fs = 44100
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait for recording to finish
    write(filename, fs, audio)  # Save recording as WAV file
    print("Audio recording saved.")
    # display(Audio(filename, autoplay=True))
    return filename


################### Delete audio 
# def delete_recording():
#     return 

################### Speech to text START
def speech_2_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return(result["text"])

# audio_path = "./asset/HODL Whisper 1.m4a"
# result = speech_2_text(audio_path)

################### ChatGPT API Call
# Function to call chatGPT API with a prompt of your choice 
def chatGPTCall(prompt):
    cmd = f"""
    curl https://api.openai.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{{
        "model": "gpt-3.5-turbo",
        "messages": [{{"role": "user", "content": "{prompt}"}}],
        "temperature": 0.9
    }}'
    """
    # Run the command and capture the output
    output = subprocess.check_output(cmd, shell=True)

    response_str = output.decode('utf-8')
    response_dict = json.loads(response_str)
    foutput = response_dict['choices'][0]['message']['content']
    return(foutput)


################### Google Event API access
# Access Google Events API, Input: Location; Output: A dictionary of 10 events around the area 
def get_google_events(location):
    params = {
    "engine": "google_events",
    "q": "Events in " + location,
    "hl": "en",
    "gl": "us",
    "htichips": "date:week", 
    "no_cache": False,
    "api_key": googleevents_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    events_results = results["events_results"]
    
    return events_results

def event_formatting(events_results):
    print("Hey lovely user, there are %s events happening in your vicinity, journey juice don't want you to compromise on fun."%(len(events_results)))
    add_lst = []
    title_lst = []
    link_lst = []

    for i in range(0, len(events_results)):
    #print("**********Event %s**********"%(i+1))
    
        try:
            e_title = events_results[i]['title']
        except:
            pass
        
        try:
            e_start_date = events_results[i]['date']['start_date']
        except:
            pass
        
        try:
            e_when = events_results[i]['date']['when']
        except:
            pass
        
        try:
            e_address = events_results[i]['address']
        except:
            pass
        
        try:
            e_description = events_results[i]['description']
        except:
            pass
        
        try:
            e_link = events_results[i]['link']
        except:
            pass
        
        e_address = ', '.join(e_address)  

        #print("Name = ", e_title)
        #print("Date = ", e_start_date)
        #print("Time = ", e_when)
        #print("Address = ", e_address)
        #print("Description = ", e_description)
        #print("Link = ", e_link)
        add_lst.append(e_address)
        title_lst.append(e_title)
        link_lst.append(e_link)
    return(add_lst, title_lst,link_lst)

    
################### Text to speech 
def text2speech(text, language, output_name):
    file_path = output_name + ".mp3"
    myobj = gTTS(text=text, lang=language, slow=False)
    myobj.save(file_path)
    return file_path
    

def play_audio_file(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path, format="mp3")
    # Play the audio file
    play(audio)

################### Map
def generate_map(add_lst, title_lst, link_lst):
    # Set up the Google Maps API client
    gmaps = googlemaps.Client(key=googlemap_key)

    # Define the addresses to geocode
    addresses = add_lst

    # Geocode each address to get its latitude and longitude
    locations = []
    for address in addresses:
        geocode_result = gmaps.geocode(address)[0]['geometry']['location']
        lat, lon = geocode_result['lat'], geocode_result['lng']
        locations.append((lat, lon, address))

    # Create a map centered on the first location
    m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13, tiles='Stamen Terrain')

    # Add a marker for each location
    i = 0 
    for location in locations:
        popup_text = f"<b>Location:</b> {location[2]} <br><b>Title:</b> {title_lst[i]}<br><b>Link:</b> {link_lst[i]}<br><b>Category: </b>"
        folium.Marker(location=[location[0], location[1]], popup=popup_text, icon=folium.Icon(color='green')).add_to(m)
        i = i+1

    # Display the map
    m


###################### Execution ############################################ 
{
"English" : "en",
"Hindi" : "hi",
"Mandarin" : "zh-CN"
}

### Speech to text ###
audio_path = record()
result = speech_2_text(audio_path)
# process the results to strip '
result = result.replace("'", "")

### Chatgpt to extract location ###
prompt_location_extraction = "Tell the city, state, postal code of location in the sentence. Only return in a (city, state, postal code) format. If no postal code, just return (city, state). If you do not know, say I do not know: "
print(prompt_location_extraction + result)
chatGPT_result_location = chatGPTCall(prompt_location_extraction + result) # location (city, state, postal code) 
print(chatGPT_result_location)  #output 1 -> show on the page, used in the following model

### Chatgpt to extract location name ###
prompt_location_name = "Tell me the name of the place in one word. If you do not know, say I do not know: "
print(prompt_location_name + result)
chatGPT_result_name = chatGPTCall(prompt_location_name + result) # location name: times square
print(chatGPT_result_name)  #output 2 -> not show on the page, used in the following model

### Text2Speech ###
# Call ChatGPT for travel recommendation 
travel_prompt = "Tell me about this famous travel location, its history and significance, in fifty words: " 
chatGPT_result_locationInfo = chatGPTCall(travel_prompt + chatGPT_result_name) # Location intro
print(chatGPT_result_locationInfo)  # front end 

mytext = chatGPT_result_locationInfo + "Journey Juice handpicked for you event happening nearby. Please check them out in the map. Have fun!"
language = 'en' # TODO: create dictionary to map the language code to language name provided by the user.
output_name = 'speak'
# convert text to speech and save the audio file 
audio_file = text2speech(mytext, language, output_name)
# play the saved audio 
play_audio_file(audio_file) 

### Events ###
event_output = get_google_events(chatGPT_result_location)
add_lst, title_lst, link_lst = event_formatting(event_output)
generate_map(add_lst, title_lst, link_lst)