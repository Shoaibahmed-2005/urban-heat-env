"""
Core Urban Heat Island environment logic.

The 8x8 city grid holds cells with surface type, temperature, and population
density.  An RL agent places cooling interventions under a fixed budget to
reduce heat exposure.
"""

from __future__ import annotations

import sys
import os
# Allow importing models from the project root when this module is imported
# from within the server sub-package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
from typing import Any

from models import CellState, CityState, TaskResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 8
INITIAL_BUDGET = 15.0
MAX_STEPS = 15

SURFACE_TYPES = ["road", "building", "park", "water"]

# Intervention specs: cost (budget units), radius (cells), temp_reduction (°C)
INTERVENTIONS: dict[str, dict[str, float]] = {
    "green_roof":         {"cost": 2.0, "radius": 1, "temp_reduction": 3.0},
    "reflective_surface": {"cost": 1.0, "radius": 1, "temp_reduction": 1.5},
    "tree_canopy":        {"cost": 3.0, "radius": 2, "temp_reduction": 4.0},
}

TASK_IDS = ["reduce_avg_temp", "protect_dense_zones", "full_mitigation"]


# ---------------------------------------------------------------------------
# CityGrid
# ---------------------------------------------------------------------------

class CityGrid:
    """
    Represents the 8×8 city grid for a single episode.

    Attributes
    ----------
    surface_types : list[list[str]]
        Surface type for each cell.  Kept constant across an episode.
    temperatures : np.ndarray shape (8, 8)
        Current temperature in °C for each cell.
    base_temperatures : np.ndarray shape (8, 8)
        Initial temperatures recorded at reset — used for scoring.
    population_density : np.ndarray shape (8, 8)
        Population density [0, 1] for each cell.
    budget : float
        Remaining budget for the current episode.
    step_count : int
        Number of steps taken so far.
    done : bool
        Whether the episode has ended.
    """

    def __init__(self) -> None:
        self.surface_types: list[list[str]] = []
        self.temperatures: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.base_temperatures: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.population_density: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE))
        self.budget: float = INITIAL_BUDGET
        self.step_count: int = 0
        self.done: bool = False

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int = 42) -> CityState:
        """Initialise/reinitialise the grid deterministically."""
        np.random.seed(seed)

        self.step_count = 0
        self.budget = INITIAL_BUDGET
        self.done = False

        # Generate surface types
        self.surface_types = [
            [
                np.random.choice(SURFACE_TYPES, p=[0.35, 0.40, 0.15, 0.10])
                for _ in range(GRID_SIZE)
            ]
            for _ in range(GRID_SIZE)
        ]

        # Generate base temperatures based on surface type
        surface_base: dict[str, tuple[float, float]] = {
            "road":     (35.0, 42.0),
            "building": (30.0, 40.0),
            "park":     (25.0, 32.0),
            "water":    (25.0, 28.0),
        }
        temps = np.zeros((GRID_SIZE, GRID_SIZE))
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                lo, hi = surface_base[self.surface_types[r][c]]
                temps[r, c] = np.round(np.random.uniform(lo, hi), 2)

        self.temperatures = temps.copy()
        self.base_temperatures = temps.copy()

        # Generate population density
        self.population_density = np.round(
            np.random.uniform(0.0, 1.0, (GRID_SIZE, GRID_SIZE)), 3
        )

        return self._build_city_state()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        row: int,
        col: int,
        intervention_type: str,
        task_id: str,
    ) -> tuple[CityState, float, bool, dict[str, Any]]:
        """
        Apply a cooling intervention to the grid.

        Returns
        -------
        state : CityState
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            return self._build_city_state(), 0.0, True, {"error": "episode already done"}

        # Validate intervention type
        if intervention_type not in INTERVENTIONS:
            return (
                self._build_city_state(),
                0.0,
                False,
                {"error": f"Unknown intervention: {intervention_type}"},
            )

        # Validate coordinates
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return (
                self._build_city_state(),
                0.0,
                False,
                {"error": f"Coordinates ({row},{col}) out of range"},
            )

        spec = INTERVENTIONS[intervention_type]
        cost: float = spec["cost"]
        radius: int = int(spec["radius"])
        reduction: float = spec["temp_reduction"]

        # Out-of-budget: end episode, zero reward
        if cost > self.budget:
            self.done = True
            return (
                self._build_city_state(),
                0.0,
                True,
                {"error": "Insufficient budget", "budget_remaining": self.budget},
            )

        # Deduct cost
        self.budget -= cost

        # Apply temperature reduction to target cell and neighbours within radius
        temp_before = self.temperatures.copy()
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    # Distance-weighted reduction (closer = more effect)
                    dist = max(abs(dr), abs(dc))  # Chebyshev distance
                    weight = 1.0 if dist == 0 else 1.0 / (dist + 1)
                    self.temperatures[nr, nc] = max(
                        20.0,  # lower bound for realism
                        self.temperatures[nr, nc] - reduction * weight,
                    )

        # Round temperatures
        self.temperatures = np.round(self.temperatures, 2)

        self.step_count += 1

        # Compute step reward relative to the active task
        reward = self._compute_step_reward(task_id, temp_before)

        # Check done conditions
        if self.step_count >= MAX_STEPS or self.budget <= 0:
            self.done = True

        new_state = self._build_city_state()
        info: dict[str, Any] = {
            "step": self.step_count,
            "budget_remaining": self.budget,
            "cost_paid": cost,
            "intervention": intervention_type,
            "target": {"row": row, "col": col},
        }
        return new_state, reward, self.done, info

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def grade_task(self, task_id: str) -> TaskResult:
        """Compute the final score [0, 1] for the given task."""
        if task_id == "reduce_avg_temp":
            return self._grade_reduce_avg_temp()
        elif task_id == "protect_dense_zones":
            return self._grade_protect_dense_zones()
        elif task_id == "full_mitigation":
            return self._grade_full_mitigation()
        else:
            return TaskResult(
                task_id=task_id,
                score=0.0,
                details={"error": f"Unknown task id: {task_id}"},
            )

    def _grade_reduce_avg_temp(self) -> TaskResult:
        """Easy — reduce average temperature by at least 2 °C."""
        initial_avg = float(np.mean(self.base_temperatures))
        current_avg = float(np.mean(self.temperatures))
        actual_reduction = initial_avg - current_avg
        score = float(np.clip(actual_reduction / 2.0, 0.0, 1.0))
        return TaskResult(
            task_id="reduce_avg_temp",
            score=score,
            details={
                "initial_avg_temp": round(initial_avg, 3),
                "current_avg_temp": round(current_avg, 3),
                "actual_reduction": round(actual_reduction, 3),
                "target_reduction": 2.0,
            },
        )

    def _grade_protect_dense_zones(self) -> TaskResult:
        """Medium — cool the 5 highest population-density cells by at least 1.5 °C each."""
        flat_density = self.population_density.flatten()
        flat_indices = np.argsort(flat_density)[::-1][:5]
        rows = flat_indices // GRID_SIZE
        cols = flat_indices % GRID_SIZE

        cells_cooled = 0
        cell_details = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            reduction = float(self.base_temperatures[r, c] - self.temperatures[r, c])
            cooled = reduction >= 1.5
            if cooled:
                cells_cooled += 1
            cell_details.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "density": round(float(self.population_density[r, c]), 3),
                    "temp_reduction": round(reduction, 3),
                    "cooled": cooled,
                }
            )

        score = float(cells_cooled / 5)
        return TaskResult(
            task_id="protect_dense_zones",
            score=score,
            details={
                "cells_cooled": cells_cooled,
                "target_cells": 5,
                "cells": cell_details,
            },
        )

    def _grade_full_mitigation(self) -> TaskResult:
        """Hard — composite: 40% avg_temp_reduction + 40% pop_weighted_coverage + 20% budget_efficiency."""
        # --- Component 1: avg temp reduction score (same as easy task but normalised differently)
        initial_avg = float(np.mean(self.base_temperatures))
        current_avg = float(np.mean(self.temperatures))
        avg_reduction = initial_avg - current_avg
        # Target: 3 °C for full score in hard mode
        avg_temp_score = float(np.clip(avg_reduction / 3.0, 0.0, 1.0))

        # --- Component 2: population-weighted coverage
        # Coverage = fraction of population in cooled cells (reduction >= 1 °C)
        reductions = self.base_temperatures - self.temperatures
        cooled_mask = reductions >= 1.0
        total_pop = float(np.sum(self.population_density))
        if total_pop > 0:
            covered_pop = float(np.sum(self.population_density * cooled_mask))
            pop_coverage_score = float(np.clip(covered_pop / total_pop, 0.0, 1.0))
        else:
            pop_coverage_score = 0.0

        # --- Component 3: budget efficiency
        # Budget used as a fraction of starting budget; reward using more efficiently
        budget_used = INITIAL_BUDGET - self.budget
        if budget_used > 0:
            # Efficiency = temp_reduction_per_budget_unit, normalised
            efficiency = avg_reduction / budget_used  # °C per unit
            budget_efficiency_score = float(np.clip(efficiency / 0.5, 0.0, 1.0))
        else:
            budget_efficiency_score = 0.0

        # --- Weighted composite
        score = float(
            np.clip(
                0.40 * avg_temp_score
                + 0.40 * pop_coverage_score
                + 0.20 * budget_efficiency_score,
                0.0,
                1.0,
            )
        )

        return TaskResult(
            task_id="full_mitigation",
            score=score,
            details={
                "avg_temp_reduction": round(avg_reduction, 3),
                "avg_temp_score": round(avg_temp_score, 3),
                "pop_coverage_score": round(pop_coverage_score, 3),
                "budget_efficiency_score": round(budget_efficiency_score, 3),
                "budget_used": round(float(budget_used), 3),
                "weights": {"avg_temp": 0.40, "pop_coverage": 0.40, "budget_efficiency": 0.20},
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_step_reward(self, task_id: str, temp_before: np.ndarray) -> float:
        """
        Dense step reward: change in task-relevant metric after this action.
        """
        if task_id == "reduce_avg_temp":
            # Reward = reduction in average temperature this step
            reward = float(np.mean(temp_before) - np.mean(self.temperatures))
        elif task_id == "protect_dense_zones":
            # Reward = improvement in population-weighted temperature reduction
            flat_density = self.population_density.flatten()
            top5_idx = np.argsort(flat_density)[::-1][:5]
            rows = top5_idx // GRID_SIZE
            cols = top5_idx % GRID_SIZE
            before_vals = temp_before[rows, cols]
            after_vals = self.temperatures[rows, cols]
            reward = float(np.mean(before_vals - after_vals))
        elif task_id == "full_mitigation":
            # Population-weighted temperature reduction
            pop = self.population_density
            total_pop = float(np.sum(pop))
            if total_pop > 0:
                reward = float(
                    np.sum(pop * (temp_before - self.temperatures)) / total_pop
                )
            else:
                reward = float(np.mean(temp_before - self.temperatures))
        else:
            reward = 0.0

        return round(reward, 4)

    def _build_city_state(self) -> CityState:
        """Construct the Pydantic CityState from current grid state."""
        grid: list[list[CellState]] = []
        for r in range(GRID_SIZE):
            row_cells: list[CellState] = []
            for c in range(GRID_SIZE):
                row_cells.append(
                    CellState(
                        row=r,
                        col=c,
                        surface_type=self.surface_types[r][c],
                        temperature=round(float(self.temperatures[r, c]), 2),
                        population_density=round(float(self.population_density[r, c]), 3),
                    )
                )
            grid.append(row_cells)

        return CityState(
            grid=grid,
            budget=round(self.budget, 2),
            step_count=self.step_count,
            avg_temperature=round(float(np.mean(self.temperatures)), 3),
            episode_done=self.done,
        )

    # ------------------------------------------------------------------
    # Snapshot / restore (used by tests or multi-task inference)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copyable snapshot of the current grid state."""
        return {
            "surface_types": copy.deepcopy(self.surface_types),
            "temperatures": self.temperatures.copy(),
            "base_temperatures": self.base_temperatures.copy(),
            "population_density": self.population_density.copy(),
            "budget": self.budget,
            "step_count": self.step_count,
            "done": self.done,
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore grid from a previously taken snapshot."""
        self.surface_types = copy.deepcopy(snap["surface_types"])
        self.temperatures = snap["temperatures"].copy()
        self.base_temperatures = snap["base_temperatures"].copy()
        self.population_density = snap["population_density"].copy()
        self.budget = snap["budget"]
        self.step_count = snap["step_count"]
        self.done = snap["done"]
