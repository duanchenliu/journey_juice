# -*- coding: utf-8 -*-
import os
import numpy as np

try:
    import tensorflow 
except ImportError:
    pass

# import torch
import pandas as pd
import whisper
import torchaudio

from tqdm.notebook import tqdm
from pathlib import Path


################### Speech to text 
def speech_2_text(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return(result["text"])

audio_path = "./asset/HODL Whisper 1.m4a"
result = speech_2_text(audio_path)
print(result)

###################



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