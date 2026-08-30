from dataclasses import dataclass

import mujoco
import numpy as np

from twin_sim.model import SimulationModel


@dataclass(frozen=True)
class GraspObservation:
    hand_contact_count: int
    opposing_contacts: bool
    cube_linear_speed_m_s: float
    cube_angular_speed_rad_s: float
    max_hand_actuator_force: float


class GraspMonitor:
    def __init__(
        self,
        *,
        stable_dwell_s: float = 0.15,
        max_linear_speed: float = 0.03,
        max_angular_speed: float = 0.5,
        abort_force: float = 25.0,
    ):
        values = (
            stable_dwell_s,
            max_linear_speed,
            max_angular_speed,
            abort_force,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 for value in values):
            raise ValueError("grasp monitor thresholds must be positive and finite")
        self.stable_dwell_s = float(stable_dwell_s)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.abort_force = float(abort_force)
        self.ready = False
        self.abort_reason: str | None = None
        self._stable_time = 0.0

    def reset(self) -> None:
        self.ready = False
        self.abort_reason = None
        self._stable_time = 0.0

    def update(self, observation: GraspObservation, dt_s: float) -> None:
        duration = float(dt_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if self.abort_reason is not None:
            return
        if observation.max_hand_actuator_force > self.abort_force:
            self.abort_reason = "hand force limit exceeded"
            self.ready = False
            return
        stable = (
            observation.hand_contact_count >= 2
            and observation.opposing_contacts
            and observation.cube_linear_speed_m_s <= self.max_linear_speed
            and observation.cube_angular_speed_rad_s <= self.max_angular_speed
        )
        self._stable_time = self._stable_time + duration if stable else 0.0
        self.ready = self._stable_time + 1e-12 >= self.stable_dwell_s

    def observe(self, sim: SimulationModel) -> GraspObservation:
        cube_body = sim.require_body("pick_cube")
        palm_body = sim.require_body("left_palm_link")
        hand_bodies = self._descendants(sim.model, palm_body)
        contacts: list[tuple[int, np.ndarray]] = []

        for index in range(sim.data.ncon):
            contact = sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(sim.model.geom_bodyid[geom1])
            body2 = int(sim.model.geom_bodyid[geom2])
            normal = np.asarray(contact.frame[:3], dtype=float)
            if body1 in hand_bodies and body2 == cube_body:
                contacts.append((body1, normal.copy()))
            elif body2 in hand_bodies and body1 == cube_body:
                contacts.append((body2, -normal.copy()))

        opposing = self._has_opposing_parts(contacts)
        cube_velocity = sim.data.cvel[cube_body]
        actuator_force = sim.data.actuator_force[sim.hand.actuator_ids]
        maximum_force = (
            float(np.max(np.abs(actuator_force)))
            if actuator_force.size
            else 0.0
        )
        return GraspObservation(
            hand_contact_count=len(contacts),
            opposing_contacts=opposing,
            cube_linear_speed_m_s=float(np.linalg.norm(cube_velocity[3:])),
            cube_angular_speed_rad_s=float(np.linalg.norm(cube_velocity[:3])),
            max_hand_actuator_force=maximum_force,
        )

    @staticmethod
    def _has_opposing_parts(
        contacts: list[tuple[int, np.ndarray]]
        | tuple[tuple[int, np.ndarray], ...],
    ) -> bool:
        return any(
            first_body != second_body
            and float(np.dot(first_normal, second_normal)) < -0.3
            for offset, (first_body, first_normal) in enumerate(contacts)
            for second_body, second_normal in contacts[offset + 1 :]
        )

    @staticmethod
    def _descendants(model: mujoco.MjModel, root_body: int) -> set[int]:
        descendants: set[int] = set()
        for body in range(model.nbody):
            current = body
            while current:
                if current == root_body:
                    descendants.add(body)
                    break
                current = int(model.body_parentid[current])
        return descendants
