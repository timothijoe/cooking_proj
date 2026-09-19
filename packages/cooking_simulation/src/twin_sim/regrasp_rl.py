"""Residual PPO environment for the independent regrasp micro-skill.

Actor sees proprioception and ideal tactile feedback, never object truth.
Reference phases/IK are retained. This does NOT learn motion from scratch.
"""
from collections import deque

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from twin_sim.regrasp_microskill import MicroSkillConfig, RegraspMicroSkill

PHASES = ('READY', 'APPROACH', 'ESTABLISH_SUPPORT', 'SUPPORT_RETREAT',
          'RELEASE_MOVE', 'RESEAT', 'SETTLE', 'HOLD', 'FAULT')


class RegraspRLEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, *, randomize_shape=True, cycles=3, max_steps=1400, tactile=True, support_repeats=1):
        if cycles not in (1, 2, 3):
            raise ValueError('cycles must be 1–3')
        self.support_repeats = support_repeats
        self.randomize_shape, self.cycles = randomize_shape, cycles
        self.max_steps, self.tactile = max_steps, tactile
        self.action_space = spaces.Box(-1, 1, (8,), np.float32)
        # 27 q, 27 dq, 27 error, 3 load, 12 effort, 3 wrist, 9 phase,
        # phase time and request count, 8 previous actions. Three frames.
        self.frame_dim = 118
        self.observation_space = spaces.Box(-np.inf, np.inf, (3*self.frame_dim,), np.float32)
        self.skill = None
        self.history = deque(maxlen=3)
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        shape_seed = int((options or {}).get('shape_seed', self.np_random.integers(0, 1000000)))
        config = MicroSkillConfig(seed=shape_seed, randomize_shape=self.randomize_shape, support_repeats=self.support_repeats)
        self.skill = RegraspMicroSkill(config)
        self.previous_action = np.zeros(8)
        self.filtered_action = np.zeros(8)
        self.steps = 0
        self.done = False
        self.skill.request('rl-0')
        self.history.clear()
        frame = self._frame()
        self.history.extend([frame.copy() for _ in range(3)])
        return self._obs(), {'shape_seed': shape_seed, 'controller': 'residual'}

    def _frame(self):
        s = self.skill
        obs = s.observation()
        effort_ids = [s.model.actuator(f'left_finger{f}_joint{j}_actuator').id
                      for f in (2, 3, 4) for j in range(1, 5)]
        limits = np.maximum(abs(s.model.actuator_forcerange[effort_ids]).max(axis=1), 1e-6)
        phase = np.array([s.phase == p for p in PHASES], dtype=float)
        return np.r_[obs['joint_position']/np.pi,
                     np.clip(obs['joint_velocity']/5, -5, 5),
                     np.clip(obs['target_error']/.1, -5, 5),
                     np.clip(obs['normal_load_proxy']/3, 0, 5) if self.tactile else np.zeros(3),
                     np.clip(obs['actuator_torque_proxy'][effort_ids]/limits, -1, 1),
                     obs['wrist_position_board'], phase, s.elapsed/10, s.completed/3,
                     self.previous_action].astype(np.float32)

    def _obs(self):
        return np.concatenate(self.history).astype(np.float32)

    def step(self, action):
        if self.done:
            raise RuntimeError('reset required after terminal state')
        action = np.asarray(action, dtype=float)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('action must contain eight finite values')
        action = np.clip(action, -1, 1)
        s = self.skill
        old_count, old_phase = s.completed, s.phase
        old_drift = np.linalg.norm(s.data.qpos[s.pq:s.pq+3]-s.initial_potato)
        # 100 ms-scale action smoothing, no changes to physical timestep.
        self.filtered_action += .2*(action-self.filtered_action)
        a = self.filtered_action
        s.load_target = .65+.25*a[:3]
        s.approach_scale = 1+.3*a[3]
        s.support_duration = (2.4 if self.support_repeats == 3 else 1.2)/(1+.25*a[4])
        s.release_duration = 1./(1+.25*a[5])
        # Flight shape is latched when SUPPORT_RETREAT ends: preflight checks
        # exactly the curve that will execute, not a curve changed mid-flight.
        if s.phase != 'RELEASE_MOVE':
            s.lift_height = .012+.002*a[6]
            s.dip_curl_deg = 24+4*a[7]
        info = s.step()
        self.steps += 1
        drift = np.linalg.norm(s.data.qpos[s.pq:s.pq+3]-s.initial_potato)
        speed = np.linalg.norm(s.data.qvel[s.pv:s.pv+3])
        load = s.loads()
        reward = -.01 - .03*np.sum((action-self.previous_action)**2)
        reward -= 100*max(float(drift-old_drift), 0.) + .05*float(drift/.005)
        reward -= .05*float(speed/.02) + .02*float(np.maximum(load-2., 0).sum())
        if old_phase != 'RELEASE_MOVE' and s.phase in ('SUPPORT_RETREAT', 'RESEAT', 'SETTLE'):
            reward += .015*float(np.minimum(load/.65, 1).mean())
        if old_phase in ('APPROACH', 'ESTABLISH_SUPPORT') and s.phase == 'SUPPORT_RETREAT':
            reward += 2.
        reward += 10*(s.completed-old_count)
        terminated = s.phase == 'FAULT' or s.completed >= self.cycles
        truncated = self.steps >= self.max_steps and not terminated
        if s.phase == 'FAULT' or truncated:
            reward -= 10.
        if not terminated and not truncated and s.phase == 'HOLD':
            s.request(f'rl-{s.completed}')
        self.previous_action = action.copy()
        self.history.append(self._frame())
        self.done = terminated or truncated
        info.update(is_success=bool(s.completed >= self.cycles and s.phase != 'FAULT'),
                    reward=float(reward), episode_steps=self.steps,
                    action_filtered=self.filtered_action.tolist())
        return self._obs(), float(reward), terminated, truncated, info
