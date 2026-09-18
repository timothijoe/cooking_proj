"""Anchored PIP retraction followed by synchronized three-fingertip relocation.

This module plans a geometric review animation. It does not claim force closure
or a learned policy; dynamic friction must be validated separately.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from robot_core.paths import data_root
from twin_sim.kinematics import Kinematics
from twin_sim.model import SimulationModel
from twin_sim.names import LEFT_ARM, RIGHT_ARM
from twin_sim.potato_contact import PotatoConfig, build_robot_potato_xml, potato_vertices


@dataclass
class FingerRegraspPreview:
    model: object
    times_s: np.ndarray
    qpos: np.ndarray
    phases: tuple[str, ...]
    tips_m: np.ndarray
    leads_m: np.ndarray
    tip_distances_m: np.ndarray
    left_qpos_ids: np.ndarray
    hand_qpos_ids: np.ndarray
    support_tips_m: np.ndarray
    support_distances_m: np.ndarray
    finger_clearances_m: np.ndarray
    palm_lowering_m: float
    cycle_ids: np.ndarray | None = None


class FingerSolver:
    def __init__(self, model, data, finger):
        self.model, self.data = model, data
        joints = np.array([model.joint(f"left_finger{finger}_joint{j}").id for j in range(1, 5)])
        self.qids, self.dids = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        acts = [model.actuator(f"left_finger{finger}_joint{j}_actuator").id for j in range(1, 5)]
        self.limits = model.actuator_ctrlrange[acts]
        self.tip = model.geom(f"left_finger{finger}_pad").id
        self.pip = model.body(f"left_finger{finger}_link3").id
        self.potato = model.geom("potato_collision").id

    def lead(self):
        # Forward is world -Y, toward the blade. Positive: PIP ahead of tip.
        return float(self.data.geom_xpos[self.tip, 1] - self.data.xpos[self.pip, 1])

    def solve(self, target, lead=None):
        m, d = self.model, self.data
        target = np.asarray(target)

        def error():
            mujoco.mj_fwdPosition(m, d)
            value = target-d.geom_xpos[self.tip]
            return value if lead is None else np.r_[value, lead-self.lead()]

        for _ in range(160):
            residual = error()
            if np.max(np.abs(residual)) < 2e-5:
                return
            jt = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, jt, None, self.tip)
            jac = jt[:, self.dids]
            if lead is not None:
                jp = np.zeros((3, m.nv))
                mujoco.mj_jacBody(m, d, jp, None, self.pip)
                jac = np.vstack((jac, (jt-jp)[1, self.dids]))
            delta = jac.T @ np.linalg.solve(jac@jac.T+1e-8*np.eye(len(residual)), residual)
            delta *= min(1.0, .05/max(np.max(np.abs(delta)), 1e-12))
            previous = d.qpos[self.qids].copy()
            cost = float(residual@residual)
            for fraction in (1, .5, .2, .1, .05, .01):
                d.qpos[self.qids] = np.clip(previous+fraction*delta, self.limits[:, 0], self.limits[:, 1])
                trial = error()
                if trial@trial < cost:
                    break
            else:
                d.qpos[self.qids] = previous
                break
        residual = error()
        if np.max(np.abs(residual)) > .00015:
            raise ValueError(f"Finger IK cannot satisfy contact/knuckle target: {residual*1000} mm")

    def distance(self):
        return float(mujoco.mj_geomDistance(self.model, self.data, self.potato, self.tip, .2, None))

    def seat(self, lead=None):
        """Move the tip to actual mesh contact, including its finite pad radius."""
        for _ in range(12):
            mujoco.mj_fwdPosition(self.model, self.data)
            line = np.zeros(6)
            distance = mujoco.mj_geomDistance(self.model, self.data, self.potato, self.tip, .2, line)
            if abs(distance+.00015) < .0001:
                return
            normal = line[3:]-line[:3]
            normal /= max(np.linalg.norm(normal), 1e-10)
            if distance < 0:
                normal *= -1
            target = self.data.geom_xpos[self.tip].copy()-(distance+.00015)*normal
            self.solve(target, lead)
        raise ValueError("Could not seat fingertip on potato surface")

    def seat_vertical(self, lead=None):
        """Keep finger spacing and retreat distance while finding the surface."""
        for _ in range(16):
            mujoco.mj_fwdPosition(self.model, self.data)
            line = np.zeros(6)
            distance = mujoco.mj_geomDistance(self.model, self.data, self.potato, self.tip, .2, line)
            if abs(distance+.00015) < .00005:
                return
            normal = line[3:]-line[:3]
            normal /= max(np.linalg.norm(normal), 1e-10)
            if distance < 0:
                normal *= -1
            if normal[2] < .1:
                raise ValueError("Vertical seating requires an upward-facing surface")
            target = self.data.geom_xpos[self.tip].copy()
            target[2] -= (distance+.00015)/normal[2]
            self.solve(target, lead)
        raise ValueError("Could not seat fingertip without reducing spacing")

    def seat_free(self):
        """Let an auxiliary finger find a reachable contact along its own arc.

        A nearest 3D target can be unreachable for the thumb. Minimize surface
        gap instead, leaving tangential placement free within joint limits.
        """
        m, d = self.model, self.data
        lower, upper = self.limits.T
        for _ in range(160):
            mujoco.mj_fwdPosition(m, d)
            line = np.zeros(6)
            distance = mujoco.mj_geomDistance(m, d, self.potato, self.tip, .2, line)
            error = distance+.00015
            if abs(error) < .00005:
                return
            normal = line[3:]-line[:3]
            normal /= max(np.linalg.norm(normal), 1e-10)
            if distance < 0:
                normal *= -1
            jt = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, jt, None, self.tip)
            jac = normal@jt[:, self.dids]
            old = d.qpos[self.qids].copy()
            direction = -error*jac
            blocked = ((old <= lower+1e-7)&(direction < 0)) | ((old >= upper-1e-7)&(direction > 0))
            jac[blocked] = 0
            delta = -error*jac/(jac@jac+1e-8)
            delta *= min(1., .05/max(np.max(np.abs(delta)), 1e-10))
            for fraction in (1, .5, .2, .1, .01):
                d.qpos[self.qids] = np.clip(old+fraction*delta, lower, upper)
                mujoco.mj_fwdPosition(m, d)
                if abs(self.distance()+.00015) < abs(error):
                    break
            else:
                d.qpos[self.qids] = old
                break
        raise ValueError("Auxiliary finger cannot reach a support contact")


def build_finger_regrasp_preview(*, bottom_cut_m=0.0, fingertip_back_shift_m=0.0, palm_lift_m=.016) -> FingerRegraspPreview:
    # Width 80, length along slicing/retreat axis 130, height 60 mm.
    config = PotatoConfig(radii_m=(.040, .065, .030), mass_kg=.22, bottom_cut_m=bottom_cut_m)
    height = .230-float(potato_vertices(config)[:,2].min()) if bottom_cut_m else .26
    model = mujoco.MjModel.from_xml_string(build_robot_potato_xml(config, position=(.434, -.005, height)))
    data = mujoco.MjData(model)
    sim = SimulationModel(model, data, SimulationModel._arm_indices(model, LEFT_ARM),
                          SimulationModel._arm_indices(model, RIGHT_ARM), SimulationModel._hand_indices(model))
    reference = data_root()/"recordings/recorded_hand_guarded_chop_200hz_latest.npz"
    with np.load(reference, allow_pickle=False) as recording:
        for indices, name in ((sim.left, "left_arm_target_rad"), (sim.right, "right_arm_target_rad"), (sim.hand, "left_hand_target_rad")):
            data.qpos[indices.qpos_ids] = recording[name][0]
    fingers = [FingerSolver(model, data, f) for f in (2, 3, 4)]
    for finger in fingers:
        data.qpos[finger.qids] = [.6, 0, .8, .7]
    mujoco.mj_forward(model, data)
    left_ik = Kinematics(sim, sim.left, "left_palm_tcp_site")
    reference_arm = data.qpos[sim.left.qpos_ids].copy()
    palm = left_ik.fk(reference_arm)
    # Reposition the initial palm with the contact patch to preserve reachability;
    # the arm remains fixed throughout the subsequent regrasp.
    palm[1, 3] += fingertip_back_shift_m
    palm[2, 3] += palm_lift_m  # Default is 8 mm lower than the original preview.
    lifted = left_ik.ik(palm, reference_arm)
    if not lifted.success:
        raise ValueError("Initial palm placement is unreachable")
    data.qpos[sim.left.qpos_ids] = lifted.joints_rad
    mujoco.mj_forward(model, data)
    for finger, x, y in zip(fingers, (.405, .431, .457), (-.041, -.043, -.037), strict=True):
        finger.seat(lead=.007)
        target = data.geom_xpos[finger.tip].copy()
        target[0] = x
        target[1] = y+fingertip_back_shift_m
        finger.solve(target, lead=.007)
        finger.seat_vertical(lead=.007)
    mujoco.mj_forward(model, data)
    model.geom_rgba[model.geom("left_palm_grasp_pad").id, 3] = 0
    for f in (2, 3, 4):
        model.geom_rgba[model.geom(f"left_finger{f}_pad").id] = [.15, .85, .35, .8]
    helpers = [FingerSolver(model, data, f) for f in (1, 5)]
    for helper in helpers:
        model.geom_rgba[helper.tip] = [.95, .65, .10, .8]
    helper_open = [data.qpos[f.qids].copy() for f in helpers]
    collision_groups = [[i for i in range(model.ngeom)
                         if model.geom(i).name.startswith(f"left_finger{f}_")
                         and (model.geom_contype[i] or model.geom_conaffinity[i])]
                        for f in (2, 3, 4)]
    collision_pairs = [[(g, h) for g in collision_groups[a] for h in collision_groups[b]]
                       for a, b in ((0, 1), (0, 2), (1, 2))]
    states, tips, leads, distances, phases = [], [], [], [], []
    support_tips, support_distances, clearances = [], [], []

    def record(phase):
        mujoco.mj_fwdPosition(model, data)
        states.append(data.qpos.copy())
        tips.append(np.array([data.geom_xpos[f.tip].copy() for f in fingers]))
        leads.append([f.lead() for f in fingers])
        distances.append([f.distance() for f in fingers])
        support_tips.append([data.geom_xpos[f.tip].copy() for f in helpers])
        support_distances.append([f.distance() for f in helpers])
        clearances.append([min(mujoco.mj_geomDistance(model, data, g, h, .1, None)
                               for g, h in pairs) for pairs in collision_pairs])
        phases.append(phase)

    record("HOLD")
    anchors = tips[0].copy()
    start_leads = np.array(leads[0])
    for _ in range(24):
        record("HOLD")
    for t in np.linspace(0, 1, 101)[1:]:
        s = t*t*(3-2*t)
        for i, finger in enumerate(fingers):
            finger.solve(anchors[i], (1-s)*start_leads[i]+s*.0015)
        record("KNUCKLE_BACK")
    for _ in range(30):
        record("KNIFE_CLEAR")
    before_support = data.qpos.copy()
    for helper in helpers:
        helper.seat_free()
    helper_closed = [data.qpos[f.qids].copy() for f in helpers]
    data.qpos[:] = before_support
    for t in np.linspace(0, 1, 41)[1:]:
        s = t*t*(3-2*t)
        for helper, first, last in zip(helpers, helper_open, helper_closed, strict=True):
            data.qpos[helper.qids] = (1-s)*first+s*last
        record("SUPPORT_ESTABLISH")
    for _ in range(20):
        record("SUPPORT_HOLD")
    starts = np.array([data.geom_xpos[f.tip].copy() for f in fingers])
    high = starts+[0, 0, .012]
    back = high+[0, .008, 0]
    for label, first, last in (("TRIO_LIFT", starts, high), ("TRIO_RETREAT", high, back)):
        for t in np.linspace(0, 1, 41)[1:]:
            s = t*t*(3-2*t)
            for i, finger in enumerate(fingers):
                finger.solve((1-s)*first[i]+s*last[i])
            record(label)
    before_seating = data.qpos.copy()
    for finger in fingers:
        finger.seat_vertical()
    contacts = np.array([data.geom_xpos[f.tip].copy() for f in fingers])
    data.qpos[:] = before_seating
    for t in np.linspace(0, 1, 41)[1:]:
        s = t*t*(3-2*t)
        for i, finger in enumerate(fingers):
            finger.solve((1-s)*back[i]+s*contacts[i])
        record("TRIO_PLACE")
    # Return the support fingers only after all three main fingertips re-seat.
    for t in np.linspace(0, 1, 41)[1:]:
        s = t*t*(3-2*t)
        for helper, first, last in zip(helpers, helper_closed, helper_open, strict=True):
            data.qpos[helper.qids] = (1-s)*first+s*last
        record("SUPPORT_RELEASE")
    for _ in range(40):
        record("HOLD_END")
    # Place the blade face next to the most forward PIP envelope. Withdraw
    # before fingertip lifting; this is a geometric guide, not knife force control.
    data.qpos[:] = states[0]
    mujoco.mj_forward(model, data)
    knife = model.geom("right_knife_blade").id
    right_ik = Kinematics(sim, sim.right, "right_tool_tip_site")
    right = data.qpos[sim.right.qpos_ids].copy()
    base_pose = right_ik.fk(right)
    face_y = data.geom_xpos[knife, 1] + np.abs(data.geom_xmat[knife].reshape(3, 3)[1])@model.geom_size[knife]
    cleared = 0.0
    for index, (qpos, phase) in enumerate(zip(states, phases, strict=True)):
        data.qpos[:] = qpos
        mujoco.mj_fwdPosition(model, data)
        if phase == "KNIFE_CLEAR":
            cleared = min(1, cleared+1/30)
        target = base_pose.copy()
        target[1, 3] += min(data.xpos[f.pip, 1] for f in fingers)-.0107-face_y-.035*cleared
        target[2, 3] += .045*cleared
        solved = right_ik.ik(target, right, tolerance=5e-5)
        if not solved.success:
            raise ValueError("Knife guide pose is unreachable")
        right = solved.joints_rad
        states[index][sim.right.qpos_ids] = right
    return FingerRegraspPreview(model, np.arange(len(states))*.02, np.array(states), tuple(phases),
                                np.array(tips), np.array(leads), np.array(distances),
                                sim.left.qpos_ids, sim.hand.qpos_ids,
                                np.array(support_tips), np.array(support_distances),
                                np.array(clearances), .008)
