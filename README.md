# journey_juice

## Dependencies
- PyAudio: `brew install portaudio ffmpeg`
- Backend: Flask: refer to [requirement.txt](./requirements.txt)
- Frontend: http-server: `npm install http-server`
- GoogleAPI: export GOOGLEMAP_KEY, GOOGLEEVENTS_KEY, OPENAI_API_KEY
    - NOTE: NEVER CHECK-IN ANY API KEYS!!!

## Configure
Under project root directory: journey_juice
1. `python -m venv .venv` if .venv is not create
2. `source ./.venv/bin/activate` to start venv
3. `pip install -r requirements.txt` to install all dependencies
4. Use terminal: `FLASK_APP=src/main/server.py python -m flask run` to start the backend server. Refer to [collab notebook](https://colab.research.google.com/drive/1pKlqC968zMQzcW0VMdK-RERfoSKJmgUi#scrollTo=wcJpfpg4R2hz)
5. Use a new terminal: `http-server` to start the frontend server: http://127.0.0.1:8080 
6. `deactivate` to exit curent venv

## Deploy
- Nice to have: Containerized the app (contact admin for help)
- Deploy on Azure

## TODO
- Choose language from webpage
- Error handlding for bad input: bad location
- Enable launch button to generate result
- Resize the webpage: map, text-holder, mobile-view, etc
- Containerize for deployment
