#!/bin/bash

# Get the poetry environment path
POETRY_ENV_PATH=$(poetry env info --path 2>/dev/null)
REALTTS_INIT_FILE="${POETRY_ENV_PATH}/lib/python3.10/site-packages/RealtimeTTS/__init__.py"

# Update the import line to include SafePipe
sed -i 's/from .engines import BaseEngine, TimingInfo/from .engines import BaseEngine, TimingInfo, SafePipe/' "$REALTTS_INIT_FILE"

# Update the __all__ list to include SafePipe
sed -i 's/"ZipVoiceEngine", "ZipVoiceVoice"/"ZipVoiceEngine", "ZipVoiceVoice", "SafePipe"/' "$REALTTS_INIT_FILE"


REALTTS_ENGINE_FILE="${POETRY_ENV_PATH}/lib/python3.10/site-packages/RealtimeTTS/engines/__init__.py"

# Add the import line for SafePipe
sed -i '1s/from .base_engine import BaseEngine, TimingInfo/from .base_engine import BaseEngine, TimingInfo\nfrom .safepipe import SafePipe/' "$REALTTS_ENGINE_FILE"

# Update the __all__ list to include SafePipe
sed -i 's/"ZipVoiceEngine", "ZipVoiceVoice"/"ZipVoiceEngine", "ZipVoiceVoice", "SafePipe"/' "$REALTTS_ENGINE_FILE"

# Execute the command passed to docker run
exec "$@"