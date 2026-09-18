"""First curriculum: recorded claw-hand retreat under external potato impulses.

The right arm holds its initial raised knife pose. This is NOT yet a cutting
policy or a validated grasp. Full object state is explicitly privileged input.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from robot_core.paths import data_root
from twin_sim.model import SimulationModel
from twin_sim.names import LEFT_ARM, RIGHT_ARM
from twin_sim.potato_contact import PotatoConfig, build_robot_potato_xml, hand_tactile


class PotatoRegraspEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, *, reference_path: Path | None = None, disturbance=True,
                 render_mode=None, tactile_enabled=True):
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.render_mode = render_mode
        self.disturbance = bool(disturbance)
        self.tactile_enabled = bool(tactile_enabled)
        source = reference_path or data_root() / "recordings/recorded_hand_guarded_chop_200hz_latest.npz"
        self.reference_path = Path(source)
        with np.load(source, allow_pickle=False) as record:
            self.times = np.array(record["time_s"], dtype=float)
            self.left_ref = np.array(record["left_arm_target_rad"], dtype=float)
            self.hand_ref = np.array(record["left_hand_target_rad"], dtype=float)
            self.right_ref = np.array(record["right_arm_target_rad"], dtype=float)
        n = len(self.times)
        if (n < 2 or self.times.shape != (n,) or self.times[0] != 0
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)):
            raise ValueError("reference time must start at zero and increase strictly")
        for values, width in ((self.left_ref, 7), (self.right_ref, 7), (self.hand_ref, 20)):
            if values.shape != (n, width) or not np.isfinite(values).all():
                raise ValueError("invalid recorded reference shape or values")
        # Residual left arm (7), hand (20), and reference clock rate (1).
        self.action_space = spaces.Box(-1, 1, shape=(28,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32),
            "tactile": spaces.Box(-np.inf, np.inf, shape=(80,), dtype=np.float32),
            "privileged": spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32),
            "phase": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
        })
        self._renderer = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.shape = PotatoConfig(
            radii_m=tuple(np.array([.060, .035, .028])*self.np_random.uniform(.95, 1.05, 3)),
            shape_seed=int(self.np_random.integers(0, 100000)),
        )
        self.model = mujoco.MjModel.from_xml_string(build_robot_potato_xml(self.shape))
        # Hide the non-colliding grasp guide inherited from pick-and-place;
        # otherwise it looks like a small platform beneath the potato.
        guide = self.model.geom("left_palm_grasp_pad").id
        self.model.geom_rgba[guide, 3] = 0
        self.data = mujoco.MjData(self.model)
        self.left = SimulationModel._arm_indices(self.model, LEFT_ARM)
        self.right = SimulationModel._arm_indices(self.model, RIGHT_ARM)
        self.hand = SimulationModel._hand_indices(self.model)
        self.qids = np.r_[self.left.qpos_ids, self.hand.qpos_ids]
        self.dids = np.r_[self.left.dof_ids, self.hand.dof_ids]
        self.aids = np.r_[self.left.actuator_ids, self.hand.actuator_ids]
        for indices, target in ((self.left, self.left_ref[0]), (self.hand, self.hand_ref[0]), (self.right, self.right_ref[0])):
            limits = self.model.actuator_ctrlrange[indices.actuator_ids]
            if np.any(target < limits[:, 0]-1e-6) or np.any(target > limits[:, 1]+1e-6):
                raise ValueError("reference initial pose exceeds actuator limits")
            self.data.qpos[indices.qpos_ids] = target
            self.data.ctrl[indices.actuator_ids] = target
        joint = self.model.joint("potato_free").id
        self.potato_qpos = int(self.model.jnt_qposadr[joint])
        self.potato_dof = int(self.model.jnt_dofadr[joint])
        self.potato_body = self.model.body("potato").id
        self.potato_geom = self.model.geom("potato_collision").id
        self.knife_geom = self.model.geom("right_knife_blade").id
        self.hand_geoms = set()
        palm = self.model.body("left_palm_link").id
        for g in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[g])
            while body and body != palm:
                body = int(self.model.body_parentid[body])
            if body == palm:
                self.hand_geoms.add(g)
        self.clock = 0.0
        self.steps = 0
        self.previous_action = np.zeros(28)
        self.initial_position = self.data.qpos[self.potato_qpos:self.potato_qpos+3].copy()
        self.impulse_time = float(self.np_random.uniform(1.0, 1.4))
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.push_force = np.array([np.cos(angle), np.sin(angle), 0.0]) * 2.0
        self._done = False
        # Resolve insertion overlaps under physical, finite-force servos before
        # starting the episode. No state projection or object weld is used.
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        settled = self._info()
        if settled["potato_penetration_m"] > .002 or settled["knife_hand_contact"]:
            raise RuntimeError("could not initialize a valid potato contact state")
        self.data.time = 0.0
        self.initial_position = self.data.qpos[self.potato_qpos:self.potato_qpos+3].copy()
        return self._observation(), self._info()

    def _observation(self):
        tactile = hand_tactile(self.model, self.data).ravel()
        if not self.tactile_enabled:
            tactile[:] = 0
        return {
            "proprio": np.r_[self.data.qpos[self.qids], self.data.qvel[self.dids]].astype(np.float32),
            "tactile": tactile,
            "privileged": np.r_[self.shape.radii_m,
                self.data.qpos[self.potato_qpos:self.potato_qpos+3]-self.initial_position,
                self.data.qpos[self.potato_qpos+3:self.potato_qpos+7],
                self.data.qvel[self.potato_dof:self.potato_dof+6]].astype(np.float32),
            "phase": np.array([min(self.clock/self.times[-1], 1), min(self.steps/700, 1)], dtype=np.float32),
        }

    def _info(self):
        force, knife_contact, penetration = 0.0, False, 0.0
        wrench = np.zeros(6)
        for i, contact in enumerate(self.data.contact):
            pair = {int(contact.geom1), int(contact.geom2)}
            hand_contact = bool(pair & self.hand_geoms)
            if hand_contact and self.knife_geom in pair:
                knife_contact = True
            if self.potato_geom in pair:
                penetration = max(penetration, -float(contact.dist))
                if hand_contact and contact.efc_address >= 0:
                    mujoco.mj_contactForce(self.model, self.data, i, wrench)
                    force += max(0, float(wrench[0]))
        return {"potato_hand_force_n": force, "knife_hand_contact": knife_contact,
                "potato_penetration_m": penetration,
                "displacement_m": float(np.linalg.norm(self.data.qpos[self.potato_qpos:self.potato_qpos+3]-self.initial_position)),
                "reference_phase": float(self.clock/self.times[-1]),
                "privileged_observations": True}

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (28,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("action must be 28 finite values in [-1, 1]")
        if self._done:
            raise RuntimeError("reset required before stepping")
        advance = .01 * (1 + .5*action[-1])
        self.clock = min(self.times[-1], self.clock + advance)
        reference = np.r_[[np.interp(self.clock, self.times, self.left_ref[:, j]) for j in range(7)],
                          [np.interp(self.clock, self.times, self.hand_ref[:, j]) for j in range(20)]]
        limits = self.model.actuator_ctrlrange[self.aids]
        self.data.ctrl[self.aids] = np.clip(reference + action[:27]*np.r_[np.full(7, .04), np.full(20, .15)], limits[:, 0], limits[:, 1])
        transient_knife_contact = False
        transient_penetration = 0.0
        for _ in range(5):
            self.data.xfrc_applied[self.potato_body] = 0
            if self.disturbance and self.impulse_time <= self.data.time < self.impulse_time+.08:
                self.data.xfrc_applied[self.potato_body, :3] = self.push_force
            mujoco.mj_step(self.model, self.data)
            subinfo = self._info()
            transient_knife_contact |= subinfo["knife_hand_contact"]
            transient_penetration = max(transient_penetration, subinfo["potato_penetration_m"])
        self.data.xfrc_applied[self.potato_body] = 0
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        info = self._info()
        info["knife_hand_contact"] |= transient_knife_contact
        info["potato_penetration_m"] = max(info["potato_penetration_m"], transient_penetration)
        velocity = self.data.qvel[self.potato_dof:self.potato_dof+6]
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        failed = bool(not finite or info["knife_hand_contact"] or info["displacement_m"] > .15
                      or info["potato_penetration_m"] > .008)
        completed = bool(self.clock >= self.times[-1])
        reward = float(1 - 20*info["displacement_m"] - 2*np.linalg.norm(velocity[:3])
                       - .05*np.linalg.norm(velocity[3:]) + .1*min(info["potato_hand_force_n"], 3)
                       - .02*max(0, info["potato_hand_force_n"]-10)**2
                       - .01*np.sum((action-self.previous_action)**2)) if finite else -100.0
        if failed:
            reward -= 20
        self.previous_action = action.copy()
        terminated = failed or completed
        truncated = bool(self.steps >= 700 and not terminated)
        info["is_success"] = bool(completed and not failed and info["displacement_m"] < .02 and info["potato_hand_force_n"] > .2)
        info["termination_reason"] = "failure" if failed else "reference_complete" if completed else "time_limit" if truncated else "running"
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.435, -.015, .285]
        camera.distance = .5
        camera.azimuth = 60
        camera.elevation = -28
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
