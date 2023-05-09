# -*- coding: utf-8 -*-
import os
import numpy as np
import json
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
from pathlib import Path
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import time
import os.path
import json
import requests

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

app = Flask(__name__)
CORS(app)


# get API KEY environmental variables
googlemap_key = os.environ.get('GOOGLEMAP_KEY')
googleevents_key = os.environ.get('GOOGLEEVENTS_KEY')

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
    # Replace with your actual API key
    
    url = "https://api.openai.com/v1/chat/completions"
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    output = response.json()
    
    # Run the command and capture the output
    print("Calling chatGPT API...")
    
    try:
        foutput = output['choices'][0]['message']['content']
    except KeyError:
        return
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
    try:
        events_results = results["events_results"] # TODO: handle invalid input
    except KeyError:
        return []
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
def geocode_map(add_lst, title_lst, link_lst):
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

    return locations
    # Create a map centered on the first location
    # m = folium.Map(location=[locations[0][0], locations[0][1]], zoom_start=13, tiles='Stamen Terrain')

    # # Add a marker for each location
    # i = 0
    # for location in locations:
    #     popup_text = f"<b>Location:</b> {location[2]} <br><b>Title:</b> {title_lst[i]}<br><b>Link:</b> {link_lst[i]}<br><b>Category: </b>"
    #     folium.Marker(location=[location[0], location[1]], popup=popup_text, icon=folium.Icon(color='green')).add_to(m)
    #     i = i+1

    # # Display the map
    # m


###################### Execution ############################################
def output(chatGPT_result_name, language):
    ### Text2Speech ###
    travel_prompt = f"In {language} language. Tell me about {chatGPT_result_name}, its history and significance, in one paragraph. If you do not know, please say this: please reenter a valid location."
    chatGPT_result_locationInfo = chatGPTCall(travel_prompt) # Location intro
    print(chatGPT_result_locationInfo)
    if not chatGPT_result_locationInfo:
        return

    # # convert text to speech and save the audio file
    # audio_file = text2speech(mytext, language, output_name)
    # # play the saved audio
    # play_audio_file(audio_file)
    audio = {
        "text": chatGPT_result_locationInfo
    }
    return audio

def main(result):
    {
    "English" : "en",
    "Hindi" : "hi",
    "Mandarin" : "zh-CN"
    }

    # ### Speech to text ###
    # audio_path = record()
    # result = speech_2_text(audio_path)
    # # process the results to strip '
    # result = result.replace("'", "")

    ### Chatgpt to extract location ###
    prompt_location_extraction = "Tell the city, state, postal code of location in the sentence. Only return in a (city, state, postal code) format. If no postal code, just return (city, state). If you do not know, say I do not know: "
    print(prompt_location_extraction + result)
    chatGPT_result_location = chatGPTCall(prompt_location_extraction + result) # location (city, state, postal code)
    print(chatGPT_result_location)  #output 1 -> show on the page, used in the following model
    if not chatGPT_result_location:
        return

    ### Chatgpt to extract location name ###
    prompt_location_name = "Tell me the name of the place in a short phrase no longer than 5 words. If you do not know, say I do not know. You can either say the name of the place or I do not know: "
    print(prompt_location_name + result)
    chatGPT_result_name = chatGPTCall(prompt_location_name + result) # location name: times square
    print(chatGPT_result_name)  #output 2 -> not show on the page, used in the following model
    if not chatGPT_result_name:
        return

    ### Events ###
    event_output = get_google_events(chatGPT_result_location)
    add_lst, title_lst, link_lst = event_formatting(event_output)
    locations = geocode_map(add_lst, title_lst, link_lst)
    event = {
        'add_lst': add_lst,
        'title_lst': title_lst,
        'link_lst': link_lst,
        'locations': locations
    }
    return {
        "event": event,
        "chatGPT_result_name": chatGPT_result_name
    }

@app.route('/transcript', methods=['POST'])
def transcript():
    with open('./result.json', 'w') as f:
        json.dump({}, f)
    data = request.json
    transcript = data['transcript']
    response = main(transcript)
    if not response:
        abort(400, description='Bad input: audio field is missing or invalid')
    response['transcript'] = transcript

    with open("./result.json", "w") as result:
        try:
            result.write(response.to_json())
        except AttributeError:
            result.write(json.dumps(response))
    return jsonify(response)

@app.route('/result', methods=['POST'])
def result():
    data = request.json
    language = data['language']
    response = {}
    start_time = time.time()
    try:
        with open('./result.json', 'r') as f:
            while len(response) == 0 and time.time() - start_time < 5:
                time.sleep(1)
                try:
                    response = json.load(f)
                    chatGPT_result_name = response["chatGPT_result_name"]
                    text = output(chatGPT_result_name, language)
                    response["audio"] = text
                except KeyError:
                    pass
                except json.decoder.JSONDecodeError:
                    pass
                    # abort(400, description='Bad input: audio field is missing or invalid')
                print("still loading...")
    except FileNotFoundError:
        abort(400, description='Bad input: audio field is missing or invalid')
    # if len(response) > 0:
    #     with open('./result.json', 'w') as f:
    #         json.dump({}, f)
    return jsonify(response)

if __name__ == '__main__':
    app.run()
