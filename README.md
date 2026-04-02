# Urban Heat Island Mitigation Planner

This repository implements the `urban_heat_env`, an RL environment created for the OpenEnv Hackathon.

## Goal
An RL agent acts as a city planner. It evaluates an 8x8 city grid containing surface types, temperatures, and population densities. Under a fixed budget, the agent places cooling interventions (green roofs, reflective surfaces, tree canopies) to reduce heat exposure.

## Details
- **Action Space**:
    - `intervention_type`: What to place (`green_roof`, `reflective_surface`, or `tree_canopy`).
    - `row` (0-7), `col` (0-7): The location on the grid.
- **Observation Space**: An 8x8 grid composed of `CellState` objects (each holding surface_type, temperature, and population_density), plus `budget` and `step_count`.
- **Tasks**:
    - **reduce_avg_temp (Easy)**: Reduce average grid temperature by at least 2°C.
    - **protect_dense_zones (Medium)**: Cool the 5 highest population-density cells by at least 1.5°C each.
    - **full_mitigation (Hard)**: Composite score based on average temperature reduction, population-weighted coverage, and budget efficiency.

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Copy the example environment variables file and insert your Hugging Face API token:
   ```bash
   cp .env.example .env
   ```
   Open `.env` to configure your `HF_TOKEN`, the correct `MODEL_NAME`, etc.

## Running the Components

### 1. Start the Environment Server

Host the FastAPI server on port 7860:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 2. Run Inference

In a separate terminal, launch the agent loop:
```bash
python inference.py
```
This script queries the environment endpoints, uses the LLM via standard OpenAI-compatible completions to pick actions, and prints the rewards out to standard output before reporting on final scenario task scores.

### Docker Support

Alternatively, build and run using Docker:
```bash
docker build -t urban-heat-env .
docker run -p 7860:7860 urban-heat-env
```
