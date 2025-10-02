# Use CUDA base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HUGGINGFACE_HUB_CACHE=/cache \
    POETRY_VERSION=2.1.3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    libasound2 \
    alsa-utils \
    libasound2-dev \
    alsa-utils \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Poetry
RUN pip install poetry==${POETRY_VERSION}

# Set up project directory
WORKDIR /app

# Copy only dependency files first for caching
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry config virtualenvs.create true && \
    poetry install --no-interaction --no-root && \
    poetry env info

# Download model weights (using build-time secret for HF token)
ARG HF_TOKEN
RUN mkdir -p checkpoints/openaudio-s1-mini
RUN --mount=type=secret,id=hf_token \
    pip install huggingface-hub && \
    huggingface-cli download fishaudio/openaudio-s1-mini \
    --local-dir checkpoints/openaudio-s1-mini \
    --token ${HF_TOKEN}

# Copy the entrypoint script first and set permissions
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8765

# Set entrypoint to run the entrypoint.sh script before any command
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["poetry", "run", "python", "src/main.py"]