---
title: Urban Heat Env
emoji: 🌡️
colorFrom: red
colorTo: green
sdk: docker
pinned: false
---

# Urban Heat Island Mitigation Planner

An RL environment for the Meta × HuggingFace OpenEnv Hackathon.
An agent acts as a city planner placing cooling interventions on an 8x8 city grid to reduce urban heat exposure.

## Action Space
- `intervention_type`: `green_roof`, `reflective_surface`, or `tree_canopy`
- `row`: 0–7
- `col`: 0–7

## Observation Space
8x8 grid of `CellState` objects — each has `surface_type`, `temperature`, `population_density` — plus `budget` and `step_count`.

## Tasks
- **reduce_avg_temp** (Easy): Reduce average grid temperature by at least 2°C. Score = min(reduction/2.0, 1.0)
- **protect_dense_zones** (Medium): Cool the 5 highest population-density cells by at least 1.5°C each. Score = cells_cooled/5
- **full_mitigation** (Hard): Composite — 40% avg temp reduction + 40% population coverage + 20% budget efficiency

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in HF_TOKEN, API_BASE_URL, MODEL_NAME in .env
```

## Run Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
python inference.py
```

## Docker
```bash
docker build -t urban-heat-env .
docker run -p 7860:7860 urban-heat-env
```

## Environment Variables
- `API_BASE_URL` = `https://router.huggingface.co/v1`
- `MODEL_NAME` = `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` = your Hugging Face token
```