# journey_juice

## Dependencies
- PyAudio: `brew install portaudio ffmpeg`
- Backend: Flask: `FLASK_APP=src/main/server.py python -m flask run`
- Frontend: http-server: `npm install http-server`
- GoogleAPI: export GOOGLEMAP_KEY, GOOGLEEVENTS_KEY, OPENAI_API_KEY
    - NOTE: NEVER CHECK-IN ANY API KEYS!!!

## Configure
Under project root directory: journey_juice
1. `python -m venv .venv` if .venv is not create
2. `source ./.venv/bin/activate` to start venv
3. `pip install -r requirements.txt` to install all dependencies
4. `python ./src/main/journeyjuice.py` to run your code from [your notebook](https://colab.research.google.com/drive/1pKlqC968zMQzcW0VMdK-RERfoSKJmgUi#scrollTo=wcJpfpg4R2hz)
5. `deactivate` to exit curent venv

## Deploy
- Nice to have: Containerized the app (contact admin for help)
- Deploy on Azure

## TODO
- Choose language from webpage
- Enable launch button to generate result
- Resize the webpage: map, text-holder, mobile-view, etc
- Containerize for deployment
