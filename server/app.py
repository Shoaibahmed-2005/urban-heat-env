from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import sys
import os

# Allow importing from the root directory so Pydantic models can be loaded.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import CityState, PlacementAction, ObsResult, TaskResult, TaskDefinition
from server.environment import CityGrid, TASK_IDS

app = FastAPI(title="Urban Heat Island Mitigation Planner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
grid = CityGrid()

# Initialise grid so /state doesn't error before /reset
grid.reset()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset", response_model=CityState)
def reset():
    return grid.reset()

@app.post("/step", response_model=ObsResult)
def step(action: PlacementAction):
    new_state, reward, done, info = grid.step(
        row=action.row,
        col=action.col,
        intervention_type=action.intervention_type,
        task_id=action.task_id
    )
    return ObsResult(state=new_state, reward=reward, done=done, info=info)

@app.get("/state", response_model=CityState)
def state():
    if len(grid.surface_types) == 0:
        grid.reset()
    return grid._build_city_state()

@app.get("/tasks", response_model=list[TaskDefinition])
def get_tasks():
    return [
        TaskDefinition(
            id="reduce_avg_temp",
            difficulty="easy",
            description="Reduce the average grid temperature by at least 2°C"
        ),
        TaskDefinition(
            id="protect_dense_zones",
            difficulty="medium",
            description="Cool the 5 highest population-density cells by at least 1.5°C each"
        ),
        TaskDefinition(
            id="full_mitigation",
            difficulty="hard",
            description="Composite score: temperature reduction, population coverage, and budget efficiency"
        )
    ]

@app.get("/grade/{task_id}", response_model=TaskResult)
def grade(task_id: str):
    if task_id not in TASK_IDS:
        raise HTTPException(status_code=404, detail="Task not found")
    return grid.grade_task(task_id)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
