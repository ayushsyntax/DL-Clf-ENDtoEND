# Setup Guide

## Local WSL2 Setup
1. Install WSL2 Ubuntu, NVIDIA drivers for GPU.
2. Create venv: `python -m venv .venv`
3. Activate: `source .venv/bin/activate`
4. Install deps: `pip install -r requirements.txt`
5. Set .env from .env.example.
6. Install pre-commit: `pre-commit install`

## GPU Check
`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

## Run Training
`python scripts/run_training.py`

## Run Streamlit
`python scripts/run_streamlit.py`

## Docker Local
`docker-compose up`
Access MLflow at localhost:5000, Streamlit at 8501.
