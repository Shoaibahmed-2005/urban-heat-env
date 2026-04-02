from pydantic import BaseModel
from typing import Any


class CellState(BaseModel):
    row: int
    col: int
    surface_type: str
    temperature: float
    population_density: float


class CityState(BaseModel):
    grid: list[list[CellState]]
    budget: float
    step_count: int
    avg_temperature: float
    episode_done: bool


class PlacementAction(BaseModel):
    task_id: str
    row: int
    col: int
    intervention_type: str  # "green_roof", "reflective_surface", "tree_canopy"


class TaskResult(BaseModel):
    task_id: str
    score: float  # 0.0 to 1.0
    details: dict[str, Any]


class ObsResult(BaseModel):
    state: CityState
    reward: float
    done: bool
    info: dict[str, Any]


class TaskDefinition(BaseModel):
    id: str
    difficulty: str
    description: str
