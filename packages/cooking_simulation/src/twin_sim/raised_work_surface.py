"""Move the guarded-chop board and object as one coherent work surface."""

from dataclasses import dataclass

import mujoco
import numpy as np

from twin_sim.model import SimulationModel


@dataclass(frozen=True)
class WorkSurfaceSnapshot:
    board_geom_id: int
    board_position: np.ndarray
    cube_body_id: int
    cube_body_position: np.ndarray


def apply_work_surface_offset(
    simulation: SimulationModel,
    offset_m: float,
) -> WorkSurfaceSnapshot:
    offset = float(offset_m)
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("work surface offset must be non-negative and finite")
    board = simulation.require_geom("chopping_board")
    cube_body = simulation.require_body("guarded_chop_cube_body")
    snapshot = WorkSurfaceSnapshot(
        board,
        simulation.model.geom_pos[board].copy(),
        cube_body,
        simulation.model.body_pos[cube_body].copy(),
    )
    simulation.model.geom_pos[board, 2] += offset
    simulation.model.body_pos[cube_body, 2] += offset
    mujoco.mj_forward(simulation.model, simulation.data)
    return snapshot


def restore_work_surface(
    simulation: SimulationModel,
    snapshot: WorkSurfaceSnapshot,
) -> None:
    simulation.model.geom_pos[snapshot.board_geom_id] = snapshot.board_position
    simulation.model.body_pos[snapshot.cube_body_id] = snapshot.cube_body_position
    mujoco.mj_forward(simulation.model, simulation.data)


def work_surface_height_m(simulation: SimulationModel) -> float:
    board = simulation.require_geom("chopping_board")
    return float(
        simulation.data.geom_xpos[board, 2] + simulation.model.geom_size[board, 2]
    )
