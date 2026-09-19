"""Independent, contact-driven reference micro-skill (not a trained policy).

Only reset initializes physical qpos. All online IK uses separate scratch data.
Contact feedback is an ideal normal-load proxy, not calibrated Wuji effort.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robot_core.paths import data_root
from twin_sim.finger_regrasp import FingerSolver
from twin_sim.kinematics import Kinematics
from twin_sim.model import SimulationModel
from twin_sim.names import LEFT_ARM
from twin_sim.potato_contact import PotatoConfig, build_robot_potato_xml, potato_vertices, read_contact_patch


def build_microskill_xml(shape):
    height = .230 - potato_vertices(shape)[:, 2].min()
    root = ET.fromstring(build_robot_potato_xml(shape, position=(.434, -.005, height)))
    root.set('model', 'Left hand regrasp micro-skill')
    # Remove the entire arm and its referring objects, not just its appearance.
    for parent in list(root.iter()):
        for child in list(parent):
            if any(str(v).startswith(('right_', 'act_right_', 'guarded_chop_')) for v in child.attrib.values()):
                parent.remove(child)
    return ET.tostring(root, encoding='unicode')


@dataclass(frozen=True)
class MicroSkillConfig:
    seed: int = 0
    randomize_shape: bool = False
    board_height_estimate_m: float = .230
    retreat_m: float = .008
    support_repeats: int = 1

    def __post_init__(self):
        if self.support_repeats not in (1, 3):
            raise ValueError('support_repeats must be 1 or 3')
        if not np.isfinite(self.board_height_estimate_m) or not .227 <= self.board_height_estimate_m <= .233:
            raise ValueError('board estimate must be within 3 mm of the calibrated plane')
        if not np.isfinite(self.retreat_m) or not .004 <= self.retreat_m <= .008:
            raise ValueError('retreat must be 4–8 mm')


class RegraspMicroSkill:
    dt = .02
    busy_phases = {'APPROACH', 'ESTABLISH_SUPPORT', 'SUPPORT_RETREAT', 'RELEASE_MOVE', 'RESEAT', 'SETTLE'}

    def __init__(self, config=MicroSkillConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        scale = rng.uniform(.95, 1.05, 3) if config.randomize_shape else np.ones(3)
        self.shape = PotatoConfig(radii_m=tuple(np.array([.040, .065, .030])*scale),
                                  mass_kg=.22*float(np.prod(scale)), bottom_cut_m=.003,
                                  shape_seed=config.seed if config.randomize_shape else 0)
        self.model = m = mujoco.MjModel.from_xml_string(build_microskill_xml(self.shape))
        self.data, self.scratch = mujoco.MjData(m), mujoco.MjData(m)
        self.arm = SimulationModel._arm_indices(m, LEFT_ARM)
        self.hand = SimulationModel._hand_indices(m)
        sim = SimpleNamespace(model=m, data=self.scratch, left=self.arm,
                              require_site=lambda name: m.site(name).id)
        self.ik = Kinematics(sim, self.arm, 'left_palm_tcp_site')
        self.fingers = [FingerSolver(m, self.scratch, f) for f in (2, 3, 4)]
        self.qids = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        self.dids = m.jnt_dofadr[m.actuator_trnid[:, 0]]
        self.pq = int(m.jnt_qposadr[m.joint('potato_free').id])
        self.pv = int(m.jnt_dofadr[m.joint('potato_free').id])
        for f in self.fingers:
            ids = [m.actuator(f'left_finger{f.finger}_joint{j}_actuator').id for j in range(1, 5)]
            gain_scale = 3.5 if config.support_repeats == 3 else 2.5
            m.actuator_gainprm[ids, 0] *= gain_scale
            m.actuator_biasprm[ids, 1] *= gain_scale
            m.actuator_biasprm[ids, 2] *= np.sqrt(gain_scale)
            m.geom_rgba[f.tip] = [.15, .85, .35, 1]
        m.geom_rgba[m.geom('left_palm_grasp_pad').id, 3] = 0
        self.reset()

    def reset(self):
        m, d, k = self.model, self.data, self.scratch
        mujoco.mj_resetData(m, d)
        with np.load(data_root()/'recordings/recorded_hand_guarded_chop_200hz_latest.npz', allow_pickle=False) as rec:
            d.qpos[self.arm.qpos_ids] = rec['left_arm_target_rad'][0]
            d.qpos[self.hand.qpos_ids] = rec['left_hand_target_rad'][0]
        k.qpos[:] = d.qpos
        for f in self.fingers:
            k.qpos[f.qids] = [.6, 0, .8, .7]
        self.palm = self.ik.fk(k.qpos[self.arm.qpos_ids])
        self.palm[1, 3] += .024
        self.palm[2, 3] += .046
        result = self.ik.ik(self.palm, k.qpos[self.arm.qpos_ids])
        if not result.success:
            raise ValueError('unreachable initial wrist')
        k.qpos[self.arm.qpos_ids] = result.joints_rad
        mujoco.mj_forward(m, k)
        self.targets = np.array([[.403, -.017, .326], [.431, -.019, .326], [.457, -.013, .326]])
        for f, target in zip(self.fingers, self.targets):
            f.solve(target, dip_angle=np.radians(30))
        d.qpos[self.qids] = k.qpos[self.qids]
        d.ctrl[:] = d.qpos[self.qids]
        mujoco.mj_forward(m, d)
        if any(f.distance() < .005 for f in self.fingers):
            raise ValueError('initial fingers must be clear of the potato')
        for _ in range(250):
            mujoco.mj_step(m, d)
        self.initial_potato = d.qpos[self.pq:self.pq+3].copy()
        self.phase, self.reason = 'READY', ''
        self.elapsed = self.stable = 0.
        self.completed = 0
        self.wrist_retreats = 0
        self.support_segment_done = 0
        self.seen_requests = set()
        self.events = []
        self.max_drift = 0.
        self.cancelled = False
        self.first_support = None
        # Default reference settings; a learning controller may adjust these
        # bounded parameters without changing the physical integration period.
        self.load_target = np.full(3, .65)
        self.approach_scale = 1.
        self.support_duration = 2.4 if self.config.support_repeats == 3 else 1.2
        self.release_duration = 1.
        self.lift_height = .012
        self.dip_curl_deg = 24.
        self.phase_clock = 0.

    def loads(self):
        return np.array([read_contact_patch(self.model, self.data, [f.tip]).normal_force_n for f in self.fingers])

    def _phase(self, phase):
        self.phase, self.elapsed, self.stable = phase, 0., 0.
        self.phase_clock = 0.
        self.events.append({'time': float(self.data.time), 'phase': phase})

    def request(self, request_id):
        if request_id in self.seen_requests:
            return 'DUPLICATE'
        self.seen_requests.add(request_id)
        if self.phase in self.busy_phases:
            return 'BUSY'
        if self.phase not in ('READY', 'HOLD'):
            return 'FAULT'
        if self.completed >= 3:
            return 'WORKSPACE_LIMIT'
        self.cancelled = False
        if np.all(self.loads() > .1):
            self._start_support()
        else:
            self._phase('APPROACH')
        return 'ACCEPTED'

    def cancel(self):
        if self.phase in ('RELEASE_MOVE', 'RESEAT'):
            self.cancelled = True
            self._phase('RESEAT')
        elif self.phase in self.busy_phases:
            self._phase('HOLD')
            self.reason = 'cancelled'

    def _start_support(self):
        self.support_tips = np.array([self.data.geom_xpos[f.tip].copy() for f in self.fingers])
        self.start_potato = self.data.qpos[self.pq:self.pq+3].copy()
        self.start_palm = self.palm.copy()
        self.support_segment_done = 0
        self.targets = self.support_tips.copy()
        if self.first_support is None:
            self.first_support = float(self.data.time)
        self._phase('SUPPORT_RETREAT')

    def observation(self):
        """Deployable layout; load is an ideal tactile stand-in, effort uncalibrated."""
        return {'joint_position': self.data.qpos[self.qids].copy(),
                'joint_velocity': self.data.qvel[self.dids].copy(),
                'target_error': self.data.ctrl-self.data.qpos[self.qids],
                'normal_load_proxy': self.loads(),
                'actuator_torque_proxy': self.data.actuator_force.copy(),
                'wrist_position_board': self.data.site_xpos[self.model.site('left_palm_tcp_site').id]-[0, 0, self.config.board_height_estimate_m]}

    def _solve_goal(self, dip):
        k, d, m = self.scratch, self.data, self.model
        k.qpos[:] = d.qpos
        k.qpos[self.qids] = d.ctrl
        result = self.ik.ik(self.palm, k.qpos[self.arm.qpos_ids])
        if not result.success:
            raise ValueError('wrist IK unreachable')
        k.qpos[self.arm.qpos_ids] = result.joints_rad
        for f, target in zip(self.fingers, self.targets):
            f.solve(target, dip_angle=dip)
        return k.qpos[self.qids].copy()

    def _command(self, dip):
        goal = self._solve_goal(dip)
        d, m = self.data, self.model
        # Limit per-control-step joint motion; physical state is never assigned.
        d.ctrl[:] = np.clip(d.ctrl + np.clip(goal-d.ctrl, -.035, .035), *m.actuator_ctrlrange.T)

    def _release_pose(self, u):
        s = u*u*u*(10-15*u+6*u*u)
        self.targets = self.release_targets.copy()
        self.targets[:, 1] += self.config.retreat_m*s
        self.targets[:, 2] += self.lift_height*np.sin(np.pi*u)**2
        self.palm = self.release_palm.copy()
        self.palm[1, 3] += self.config.retreat_m*.5*s
        self.palm[2, 3] += .005*np.sin(np.pi*u)**2
        return np.radians(30+self.dip_curl_deg*np.sin(np.pi*u)**2)

    def _preflight_release(self):
        """Check the full nominal flight before lifting; no physical mutation."""
        targets, palm = self.targets.copy(), self.palm.copy()
        try:
            for u in np.linspace(0, 1, 51):
                self._solve_goal(self._release_pose(u))
        finally:
            self.targets, self.palm = targets, palm

    def step(self):
        loads = self.loads()
        dip = np.radians(30)
        if self.phase in self.busy_phases:
            self.elapsed += self.dt
            if self.elapsed > 10:
                self.reason = 'phase_timeout'
                self._phase('FAULT')
            elif self.phase in ('APPROACH', 'ESTABLISH_SUPPORT', 'RESEAT', 'SETTLE'):
                # Individual load regulation: contacted fingers stop descending.
                dz = np.clip((loads-self.load_target)*.00015*self.approach_scale, -.00012, .00012)
                self.targets[:, 2] += dz
                if self.phase == 'APPROACH':
                    self.palm[2, 3] -= .00010*self.approach_scale
                    if np.any(loads > .1):
                        self._phase('ESTABLISH_SUPPORT')
                if np.min(self.targets[:, 2]) < self.config.board_height_estimate_m+.016:
                    self.reason = 'descent_boundary'
                    self._phase('FAULT')
                else:
                    self.stable = self.stable+self.dt if np.all(loads > .1) and np.max(loads) < 3 else 0.
                    if self.stable >= .12:
                        if self.phase in ('APPROACH', 'ESTABLISH_SUPPORT'):
                            self._start_support()
                        elif self.phase == 'RESEAT':
                            self._phase('SETTLE')
                        elif self.phase == 'SETTLE':
                            if np.linalg.norm(self.data.qvel[self.pv:self.pv+3]) > .02:
                                self.stable = 0.
                                return self._integrate()
                            actual = np.array([self.data.geom_xpos[f.tip] for f in self.fingers])
                            delta = (actual[:, 1]-self.support_tips[:, 1])-(self.data.qpos[self.pq+1]-self.start_potato[1])
                            if self.cancelled:
                                self.reason = 'cancelled'
                                self._phase('HOLD')
                            elif np.all(abs(delta-self.config.retreat_m) < .003) and self.max_drift < .005:
                                self.completed += 1
                                self._phase('HOLD')
                                self.reason = 'succeeded'
                            else:
                                self.reason = 'landing_error_or_object_drift'
                                self._phase('FAULT')
            elif self.phase == 'SUPPORT_RETREAT':
                self.phase_clock += self.dt/self.support_duration
                u = min(self.phase_clock, 1.)
                if self.config.support_repeats == 3:
                    # Three advances, each followed by a short stationary hold.
                    # Preserve total displacement (4 mm), not three times it.
                    segment = min(int(u*3), 2)
                    local_u = min(max((u*3-segment)/.75, 0.), 1.)
                    smooth = local_u**3*(10-15*local_u+6*local_u**2)
                    s = (segment+smooth)/3
                    reached = int(sum(u >= (j+.75)/3-1e-10 for j in range(3)))
                else:
                    s = u*u*u*(10-15*u+6*u*u)
                    reached = int(u >= 1.)
                if reached > self.support_segment_done:
                    self.wrist_retreats += reached-self.support_segment_done
                    self.support_segment_done = reached
                    self.events.append({'time': float(self.data.time), 'phase': 'WRIST_RETREAT_DONE',
                                        'count': self.wrist_retreats, 'round': self.completed+1})
                self.palm = self.start_palm.copy()
                self.palm[1, 3] += self.config.retreat_m*.5*s
                self.targets[:, 2] += np.clip((loads-self.load_target)*.00010, -.00008, .00008)
                if u >= 1:
                    self.release_targets = self.targets.copy()
                    self.release_palm = self.palm.copy()
                    try:
                        self._preflight_release()
                        self._phase('RELEASE_MOVE')
                    except ValueError as exc:
                        self.reason = 'release_preflight: '+str(exc)
                        self._phase('FAULT')
            elif self.phase == 'RELEASE_MOVE':
                self.phase_clock += self.dt/self.release_duration
                u = min(self.phase_clock, 1.)
                dip = self._release_pose(u)
                if u >= 1:
                    self._phase('RESEAT')
            if self.phase in self.busy_phases:
                try:
                    self._command(dip)
                except ValueError as exc:
                    self.reason = str(exc)
                    self._phase('FAULT')
        return self._integrate()

    def _integrate(self):
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        drift = float(np.linalg.norm(self.data.qpos[self.pq:self.pq+3]-self.initial_potato))
        self.max_drift = max(self.max_drift, drift)
        if not all(np.isfinite(v).all() for v in (self.data.qpos, self.data.qvel, self.data.ctrl)) or drift > .025 or np.max(self.loads()) > 8:
            self.reason = 'nonfinite_or_excess_motion_or_load'
            self._phase('FAULT')
        return self.status()

    def status(self):
        return {'phase': self.phase, 'reason': self.reason, 'completed': self.completed,
                'time_s': float(self.data.time), 'loads_n': self.loads().tolist(),
                'max_drift_mm': self.max_drift*1000,
                'first_support_time_s': self.first_support,
                'wrist_retreats': self.wrist_retreats,
                'support_segment': self.support_segment_done}
