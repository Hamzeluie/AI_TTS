# AI_TTS
this repository is for transfering text to speech.
## Repository files

**main.py**: script of main service.

**test.py**: you can test script in develop mod with this script.

**test.wav**: test wav file used in test.py

**.env(optional)**: define environment variables in this file


## parameters

**HOST**: service host that you can assign in environment variable..

**PORT**: serice port that you can assign in environment variable.


# Install .venv
you can install virtual environment with **poetry**

        poetry install --no-root
        source entrypoint.sh

# Run main.py and test.py
if you use poetry you can run main.py with:

        poetry run python main.py

or you can run the main.py on host 0.0.0.0 and port 8000 with

        poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload 
also you can test your developement main.py with running test.py 

with:

        poetry run python main.py



# Docker
you can build docker image and run a container with the image.
Docker file Expose port is **8000**

with:


        docker build --no-cache --secret id=hf_token,src=.hf_token -t tts-server .
        docker run -it --gpus all -p 8000:8000 tts-server


# Input/Output structure
input and output of the service in dictionary structure.
you can access to the service with "ws://{HOST}:{PORT}"

## Input structure:
the input in simple text.(you can see input and output structur in test.py)

## output structure:

        {
                "type": string,
                "delta": base64Encoded(utf-8),
                "item_id": integer,
        }
