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
    "api_key": "9ec617496aef876ebae248c1ff5ee771ccf9dd4b0762d1524d93f35557196949"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    events_results = results["events_results"]
    
    return events_results

def event_formatting(events_results):
    print("Hey lovely user, there are %s events happening in your vicinity, journey juice don't want you to compromise on fun."%(len(events_results)))

    for i in range(0, len(events_results)):
        print("**********Event %s**********"%(i+1))
        
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

        print("Name = ", e_title)
        print("Date = ", e_start_date)
        print("Time = ", e_when)
        print("Address = ", e_address)
        print("Description = ", e_description)
        print("Link = ", e_link)

    
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



###################### Execution ############################################ 
### Speech to text ###
audio_path = record()
result = speech_2_text(audio_path)
# process the results to strip '
result = result.replace("'", "")

### Chatgpt to extract location ###
prompt_location_extraction = "Tell the city, state, postal code of location in the sentence. Only return in a (city, state, postal code) format. If no postal code, just return (city, state). If you do not know, say I do not know: "
print(prompt_location_extraction + result)
chatGPT_result_location = chatGPTCall(prompt_location_extraction + result) # location (city, state, postal code) 
print(chatGPT_result_location) 

### Chatgpt to extract location name ###
prompt_location_name = "Tell me the name of the place in one word. If you do not know, say I do not know: "
print(prompt_location_name + result)
chatGPT_result_name = chatGPTCall(prompt_location_name + result) # location name 
print(chatGPT_result_name) 

### Events ###
event_output = get_google_events(chatGPT_result_location)
event_formatting(event_output)

### Text2Speech ###
# Call ChatGPT for travel recommendation 
travel_prompt = "Tell me about this famous travel location, its history and significance, in fifty words: " 
chatGPT_result_locationInfo = chatGPTCall(travel_prompt + chatGPT_result_name) # Location intro
print(chatGPT_result_locationInfo) 


mytext = chatGPT_result_locationInfo + "Journey Juice handpicked for you event happening nearby. Please check them out in the map. Have fun!"
language = 'en'
output_name = 'speak'
# convert text to speech and save the audio file 
audio_file = text2speech(mytext, language, output_name)
# play the saved audio 
play_audio_file(audio_file)



################### ChatGPT Extract Location START
# prompt_location_extraction = "Tell the city, state, postal code of location in the sentence. Return in a (city, state, postal code) format in a single line: "

# input_GPT = prompt_location_extraction + result

# cmd = f"""
# export OPENAI_API_KEY='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' 
# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{{
#      "model": "gpt-3.5-turbo",
#      "messages": [{{"role": "user", "content": "{input_GPT}"}}],
#      "temperature": 0.7
#    }}'
# """
# # cmd = f"""
# # curl https://api.openai.com/v1/chat/completions \
# #   -H "Content-Type: application/json" \
# #   -H "Authorization: Bearer $OPENAI_API_KEY" \
# #   -d '{{
# #      "model": "gpt-3.5-turbo",
# #      "messages": [{{"role": "user", "content": "{input_GPT}"}}],
# #      "temperature": 0.7
# #    }}'
# # """

# # Run the command and capture the output
# output = subprocess.check_output(cmd, shell=True)

# # Print the output
# print(output)

##anant's API: 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
##Gen AI API: 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


################### ChatGPT Extract Location END 


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class LibriSpeech(torch.utils.data.Dataset):
#     """
#     A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
#     It will drop the last few seconds of a very small portion of the utterances.
#     """
#     def __init__(self, split="test-clean", device=DEVICE):
#         self.dataset = torchaudio.datasets.LIBRISPEECH(
#             root=os.path.expanduser("~/.cache"),
#             url=split,
#             download=True,
#         )
#         self.device = device

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, item):
#         audio, sample_rate, text, _, _, _ = self.dataset[item]
#         assert sample_rate == 16000
#         audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
#         mel = whisper.log_mel_spectrogram(audio)
        
#         return (mel, text)

# dataset = LibriSpeech("test-clean")
# loader = torch.utils.data.DataLoader(dataset, batch_size=16)

# model = whisper.load_model("base.en")
# print(
#     f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
#     f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
# )

# # predict without timestamps for short-form transcription
# options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)

# hypotheses = []
# references = []

# for mels, texts in loader: #slow slow slow!!!
#     results = model.decode(mels, options)
#     hypotheses.extend([result.text for result in results])
#     references.extend(texts)

# """## TODO: Live Audio"""

# ## Option 1 
# from google.colab import files # TODO: update
# from IPython.display import Audio
# import io
# import soundfile as sf

# # Define the duration and filename of the recording
# duration = 5  # seconds
# filename = 'recording.wav'

# # # Record audio using the built-in microphone
# # print(f"Recording audio for {duration} seconds...")
# # #!arecord -D plughw:1,0 -d $duration -f cd -t wav $filename
# # !arecord -D default -d $duration -f cd -t wav $filename # TODO update


# # Read the audio data from the file into a BytesIO object
# audio_data = io.BytesIO()
# with open(filename, 'rb') as f:
#     audio_data.write(f.read())
# audio_data.seek(0)

# # Play back the audio data
# Audio(audio_data.getvalue(), rate=44100, autoplay=True)

# # Download the WAV file to your local machine
# files.download(filename)

# ## Option 2

# # !pip install sounddevice
# # !apt-get install libportaudio2

# import sounddevice as sd
# from scipy.io.wavfile import write
# from IPython.display import Audio, display

# duration = 5  # seconds
# filename = 'recording.wav'

# print(f"Recording audio for {duration} seconds...")
# fs = 44100
# audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
# sd.wait()  # Wait for recording to finish
# write(filename, fs, audio)  # Save recording as WAV file
# print("Audio recording saved.")

# display(Audio(filename, autoplay=True))

# import sounddevice as sd
# print(sd.query_devices())

# """Whisper API """

# import whisper

# model = whisper.load_model("base")

# # #connect to google drive
# # from google.colab import drive # TODO: udpate
# # drive.mount('/content/drive')
# from pathlib import Path
# audio_path = Path("./asset/HODL Whisper 1.m4a")
# # load audio and pad/trim it to fit 30 seconds
# audio = whisper.load_audio(audio_path)
# audio = whisper.pad_or_trim(audio)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel)
# print(f"Detected language: {max(probs, key=probs.get)}")

# # decode the audio
# options = whisper.DecodingOptions()
# result = whisper.decode(model, mel, options)

# # print the recognized text
# print(result.text)



# """ChatGPT API"""

# import subprocess

# text = "provide me 2 days itineray for a trip to boston for a 5 year old"
# cmd = f"""
# curl https://api.openai.com/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{{
#      "model": "gpt-3.5-turbo",
#      "messages": [{{"role": "user", "content": "{text}"}}],
#      "temperature": 0.7
#    }}'
# """

# # Run the command and capture the output
# output = subprocess.check_output(cmd, shell=True)

# #Print the output
# #print(output)

# import json

# #response = b'{"id":"chatcmpl-78WnM8WPpdoRTNydYkg74y5oZ6SPH","object":"chat.completion","created":1682267044,"model":"gpt-3.5-turbo-0301","usage":{"prompt_tokens":14,"completion_tokens":183,"total_tokens":197},"choices":[{"message":{"role":"assistant","content":"The Charles River is an 80-mile long river in eastern Massachusetts that flows through Boston, Cambridge, and several other towns. The river starts at Echo Lake in Hopkinton and ends in Boston Harbor. \\n\\nThe Charles River basin is a popular recreational area for boating, sailing, kayaking, and rowing. It is also home to several parks, including the Esplanade, which hosts the annual Boston Pops Fireworks Spectacular on the Fourth of July.\\n\\nThe river has a long and rich history, playing an important role in the development of Boston and Cambridge. It was named after King Charles I of England and was once a major transportation route for goods and people. In the mid-20th century, the river was heavily polluted, but significant efforts have been made to clean it up.\\n\\nToday, the Charles River is a symbol of the Boston area and is beloved by residents and visitors alike."},"finish_reason":"stop","index":0}]}\n'

# response_str = output.decode('utf-8')
# response_dict = json.loads(response_str)

# print(response_dict['choices'][0]['message']['content'])

# """**Text to Speech**"""

# # pip install gtts

# # Import the required module for text 
# # to speech conversion
# from gtts import gTTS
  
# # This module is imported so that we can 
# # play the converted audio
# import os
  
# # The text that you want to convert to audio
# mytext = 'Hi, I am Dora, talented person!'
# #mytext = 'मेरा नाम जोकर'
  
# # Language in which you want to convert
# language = 'en'
# #language = 'hi'
  
# # Passing the text and language to the engine, 
# # here we have marked slow=False. Which tells 
# # the module that the converted audio should 
# # have a high speed
# myobj = gTTS(text=mytext, lang=language, slow=False)
  
# # Saving the converted audio in a mp3 file named
# # welcome 
# myobj.save("welcome1.mp3")

# # Playing the converted file
# #os.system("mpg321 welcome.mp3")

# from IPython.display import Audio

# # Load the mp3 file into memory
# mp3_file = open('welcome1.mp3', 'rb').read()

# # Play the mp3 file
# Audio(mp3_file, autoplay=True)

# """**Events API**"""

# # pip install google-search-results

# # link to the API documentation: https://serpapi.com/google-events-api

# location = "Cambridge, MA 02139"

# from serpapi import GoogleSearch
# params = {
#   "engine": "google_events",
#   "q": "Events in " + location,
#   "hl": "en",
#   "gl": "us",
#    "htichips": "date:next_week", 
#    "no_cache": False,
#   "api_key": "9ec617496aef876ebae248c1ff5ee771ccf9dd4b0762d1524d93f35557196949"
# }



# search = GoogleSearch(params)
# results = search.get_dict()
# events_results = results["events_results"]

# events_results

# eventInfo = ['title', 'date', 'address','description','link']