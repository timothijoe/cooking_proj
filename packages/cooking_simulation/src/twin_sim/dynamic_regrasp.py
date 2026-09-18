"""Finite-force synchronized regrasp; shared plant for review and residual PPO.

Only reset writes qpos. Steps use actuators and mj_step, with an unconstrained
rigid potato. Knife follows a nonpenetrating geometric chopping reference.
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from twin_sim.finger_regrasp import build_finger_regrasp_preview, FingerSolver
from twin_sim.kinematics import Kinematics
from twin_sim.model import SimulationModel
from twin_sim.names import LEFT_ARM, RIGHT_ARM
from twin_sim.potato_contact import hand_tactile


def knife_food_vertical_clearance(model, data, blade, potato):
    """Conservative world-Z separation using transformed local bounding boxes.

    Positive means the entire blade is above the entire food. Avoid treating
    mesh distance-query artifacts as real penetration; contacts are checked too.
    """
    bounds=[]
    for geom in (blade,potato):
        rotation=data.geom_xmat[geom].reshape(3,3)
        box=model.geom_aabb[geom]
        center=data.geom_xpos[geom]+rotation@box[:3]
        extent=np.abs(rotation)@box[3:]
        bounds.append((center[2]-extent[2],center[2]+extent[2]))
    return float(bounds[0][0]-bounds[1][1])


def add_close_knife_reference(plan, gap_m=.003, cut_repeats=2, cut_depth_m=.018, continuous_knife=False):
    m, d = plan.model, mujoco.MjData(plan.model)
    sim = SimulationModel(m, d, SimulationModel._arm_indices(m, LEFT_ARM),
                          SimulationModel._arm_indices(m, RIGHT_ARM), SimulationModel._hand_indices(m))
    ik = Kinematics(sim, sim.right, 'right_tool_tip_site')
    blade = m.geom('right_knife_blade').id
    potato = m.geom('potato_collision').id
    hand = [g for g in range(m.ngeom) if m.geom(g).name.startswith('left_finger')
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    d.qpos[:] = plan.qpos[0]
    mujoco.mj_forward(m, d)
    initial = ik.fk(d.qpos[sim.right.qpos_ids].copy())
    center_z = d.geom_xpos[blade, 2]
    active = np.where(np.array(plan.phases) == ('CUT_ADVANCE' if 'CUT_ADVANCE' in plan.phases else 'KNUCKLE_BACK'))[0]
    retract_end=np.where(np.array(plan.phases)=='KNUCKLE_BACK')[0][-1]
    previous = d.qpos[sim.right.qpos_ids].copy()
    last_guided = initial.copy()
    for i, phase in enumerate(plan.phases):
        d.qpos[:] = plan.qpos[i]
        mujoco.mj_forward(m, d)
        target = initial.copy()
        # Lower blade so its flat face overlaps the guiding knuckles; make two
        # shallow down/up strokes, stopping above the uncut rigid potato.
        fraction = np.clip((i-active[0])/max(1,len(active)-1), 0, 1)
        stroke = cut_depth_m*np.sin(cut_repeats*np.pi*fraction)**2 if phase in ('CUT_ADVANCE',) or ('CUT_ADVANCE' not in plan.phases and phase=='KNUCKLE_BACK') else 0.
        clear = 0.0 if continuous_knife else np.clip((i-retract_end)/30, 0, 1)
        target[2, 3] += .367-center_z-stroke+.045*clear
        target[1, 3] += .005*fraction-.035*clear
        if clear > 0:
            target = last_guided.copy()
            target[1,3] -= .035*clear
            target[2,3] += .045*clear
        for _ in range(12):
            result = ik.ik(target, previous, tolerance=5e-5)
            if not result.success:
                raise ValueError('Close knife reference unreachable')
            previous = result.joints_rad
            d.qpos[sim.right.qpos_ids] = previous
            mujoco.mj_forward(m, d)
            gap = min(mujoco.mj_geomDistance(m,d,blade,g,.1,None) for g in hand)
            food_gap = knife_food_vertical_clearance(m,d,blade,potato)
            if clear == 0 and abs(gap-gap_m) > .00015:
                target[1,3] += gap-gap_m
                continue
            if food_gap < .001:
                target[2,3] += .001-food_gap
                continue
            break
        else:
            raise ValueError('Unable to establish knife clearance')
        if clear == 0:
            last_guided = target.copy()
        plan.qpos[i,sim.right.qpos_ids] = previous
    return plan


def park_auxiliary_fingers(plan):
    """Hold thumb and little finger in their initial, already clear poses.

    Keep the same phase durations for comparison. Collision geometries remain
    active; no virtual support or hidden contact filtering replaces these fingers.
    """
    m = plan.model
    for finger in (1, 5):
        ids = [m.jnt_qposadr[m.joint(f'left_finger{finger}_joint{j}').id] for j in range(1,5)]
        plan.qpos[:, ids] = plan.qpos[0, ids]
    names = {'SUPPORT_ESTABLISH': 'HELPERS_CLEAR_WAIT', 'SUPPORT_HOLD': 'PRE_LIFT_HOLD',
             'SUPPORT_RELEASE': 'POST_PLACE_HOLD'}
    plan.phases = tuple(names.get(phase, phase) for phase in plan.phases)
    return plan


def build_continuous_regrasp(cycles=3, contact_shift_m=.024, angle_gate=True, pip_guard=False, wrist_lift_m=0.0, wrist_retreat_m=0.0, smooth_regrasp=False, angle_threshold_deg=85.0, curl_release=False, early_pip_curl=False, support_repeats=1, knife_gap_m=None, cut_depth_m=.018, continuous_knife=False):
    """Sequential slicing references with continuous wrist/knife transitions."""
    if not isinstance(cycles,int) or not 1 <= cycles <= 3:
        raise ValueError('cycles must be an integer from 1 to 3 for the current potato')
    parts=[add_close_knife_reference(park_auxiliary_fingers(build_finger_regrasp_preview(
        bottom_cut_m=.003, fingertip_back_shift_m=contact_shift_m+.008*k,
        palm_lift_m=.016 if k==0 else .017, angle_gate=angle_gate,pip_guard=pip_guard,wrist_lift_m=wrist_lift_m,wrist_retreat_m=wrist_retreat_m,smooth_regrasp=smooth_regrasp,curl_release=curl_release,early_pip_curl=early_pip_curl,support_repeats=support_repeats)),gap_m=(.006 if curl_release else .003) if knife_gap_m is None else knife_gap_m,cut_repeats=support_repeats if support_repeats>1 else 2,cut_depth_m=cut_depth_m,continuous_knife=continuous_knife) for k in range(cycles)]
    if angle_gate:
        for part in parts:
            # No auxiliary fingers are used: their former setup/release waits
            # serve no purpose. Keep the final seating hold in each round.
            removed=['HELPERS_CLEAR_WAIT','POST_PLACE_HOLD']
            if smooth_regrasp:removed.append('PRE_LIFT_HOLD')
            if continuous_knife:removed.append('KNIFE_CLEAR')
            keep=~np.isin(part.phases,removed)
            part.qpos=part.qpos[keep];part.phases=tuple(np.asarray(part.phases)[keep])
    plan=parts[0];m=plan.model
    right=SimulationModel._arm_indices(m,RIGHT_ARM).qpos_ids
    # Hold auxiliary joints at the first clear pose across every round.
    helper_ids=[m.jnt_qposadr[m.joint(f'left_finger{f}_joint{j}').id]
                for f in (1,5) for j in range(1,5)]
    for part in parts:part.qpos[:,helper_ids]=plan.qpos[0,helper_ids]
    states=[];phases=[];cycle_ids=[]
    for cycle,part in enumerate(parts):
        if cycle:
            first=states[-1].copy();next_pose=part.qpos[0].copy()
            wrist_ready=next_pose.copy();wrist_ready[right]=first[right]
            transitions=[('WRIST_FOLLOW',first,next_pose)] if continuous_knife else [('WRIST_FOLLOW',first,wrist_ready),('KNIFE_APPROACH',wrist_ready,next_pose)]
            for label,a,b in transitions:
                for t in np.linspace(0,1,61)[1:]:
                    alpha=t*t*(3-2*t)
                    states.append((1-alpha)*a+alpha*b);phases.append(label);cycle_ids.append(cycle)
        states.extend(part.qpos.copy());phases.extend(part.phases);cycle_ids.extend([cycle]*len(part.qpos))
    if pip_guard:
        qids=m.jnt_qposadr[m.actuator_trnid[:,0]]
        smooth_states=[states[0]];smooth_phases=[phases[0]];smooth_cycles=[cycle_ids[0]]
        for a,b,phase,cycle in zip(states[:-1],states[1:],phases[1:],cycle_ids[1:]):
            count=max(1,int(np.ceil(np.max(abs(b[qids]-a[qids]))/.008)))
            for alpha in np.linspace(0,1,count+1)[1:]:
                smooth_states.append((1-alpha)*a+alpha*b);smooth_phases.append(phase);smooth_cycles.append(cycle)
        states,phases,cycle_ids=smooth_states,smooth_phases,smooth_cycles
    if continuous_knife:
        initial_data=mujoco.MjData(m);initial_data.qpos[:]=states[0]
        initial_sim=SimulationModel(m,initial_data,SimulationModel._arm_indices(m,LEFT_ARM),SimulationModel._arm_indices(m,RIGHT_ARM),SimulationModel._hand_indices(m))
        initial_ik=Kinematics(initial_sim,initial_sim.right,'right_tool_tip_site')
        pose=initial_ik.fk(states[0][right]);pose[1,3]-=.035;pose[2,3]+=.025
        solution=initial_ik.ik(pose,states[0][right],tolerance=1e-6)
        if not solution.success:raise ValueError('Initial knife approach unreachable')
        first=states[0].copy();first[right]=solution.joints_rad
        prefix=[]
        for u in np.linspace(0,1,51)[:-1]:
            a=u*u*u*(10-15*u+6*u*u);prefix.append((1-a)*first+a*states[0])
        states=prefix+states;phases=['INITIAL_APPROACH']*len(prefix)+phases;cycle_ids=[0]*len(prefix)+cycle_ids
    plan.qpos=np.asarray(states);plan.phases=tuple(phases);plan.times_s=np.arange(len(states))*.02
    plan.cycle_ids=np.asarray(cycle_ids)
    # Recompute geometry for the actual joined references, including transitions.
    d=mujoco.MjData(m);fingers=[FingerSolver(m,d,f) for f in (2,3,4)];helpers=[FingerSolver(m,d,f) for f in (1,5)]
    groups=[[g for g in range(m.ngeom) if m.geom(g).name.startswith(f'left_finger{f}_')
             and (m.geom_contype[g] or m.geom_conaffinity[g])] for f in (2,3,4)]
    tips=[];leads=[];distances=[];support_tips=[];support_distances=[];clearances=[]
    for pose in plan.qpos:
        d.qpos[:]=pose;mujoco.mj_forward(m,d)
        tips.append([d.geom_xpos[f.tip].copy() for f in fingers]);leads.append([f.lead() for f in fingers])
        distances.append([f.distance() for f in fingers]);support_tips.append([d.geom_xpos[f.tip].copy() for f in helpers])
        support_distances.append([f.distance() for f in helpers])
        clearances.append([min(mujoco.mj_geomDistance(m,d,g,h,.1,None) for g in groups[a] for h in groups[b])
                           for a,b in ((0,1),(0,2),(1,2))])
    plan.tips_m=np.array(tips);plan.leads_m=np.array(leads);plan.tip_distances_m=np.array(distances)
    plan.support_tips_m=np.array(support_tips);plan.support_distances_m=np.array(support_distances)
    plan.finger_clearances_m=np.array(clearances)
    return plan


class DynamicRegraspEnv(gym.Env):
    """Three bounded fingertip pressure residuals and one shared phase speed.

    Privileged geometry/state is available to the actor. This is a research
    baseline, not a validated cutting policy; blade remains above the potato.
    """
    def __init__(self, disturbance=False, tactile_enabled=True, contact_shift_m=.024, cycles=3, angle_gate=True, pip_guard=False, wrist_lift_m=0.0, wrist_retreat_m=0.0, smooth_regrasp=False, angle_threshold_deg=85.0, curl_release=False, landing_wrist_retreat_m=0.0, early_pip_curl=False, support_repeats=1, knife_gap_m=None, cut_depth_m=.018, regrasp_speed=1.0, continuous_knife=False):
        if not 1 <= regrasp_speed <= 2 or not .001 <= cut_depth_m <= .030:
            raise ValueError('regrasp_speed must be 1–2; cut depth 1–30 mm')
        self.knife_gap_m=(.006 if curl_release else .003) if knife_gap_m is None else float(knife_gap_m)
        if not .002 <= self.knife_gap_m <= .010:raise ValueError('knife gap must be 2–10 mm')
        self.continuous_knife=continuous_knife
        self.regrasp_speed=regrasp_speed
        self.plan = build_continuous_regrasp(cycles,contact_shift_m,angle_gate,pip_guard,wrist_lift_m,wrist_retreat_m,smooth_regrasp,angle_threshold_deg,curl_release,early_pip_curl,support_repeats,self.knife_gap_m,cut_depth_m,continuous_knife)
        if not 0 <= landing_wrist_retreat_m <= .004 or (landing_wrist_retreat_m and not curl_release):
            raise ValueError("landing wrist retreat requires curl_release and must be 0–4 mm")
        self.landing_wrist_retreat_m=landing_wrist_retreat_m
        self.support_repeats=support_repeats
        self.support_ends={k:np.where((self.plan.cycle_ids==k)&(np.array(self.plan.phases)=="CUT_ADVANCE"))[0][59::60] for k in range(cycles)} if support_repeats>1 else {}
        self.curl_release=curl_release
        self.wrist_retreat_m=wrist_retreat_m
        self.angle_gate_enabled=angle_gate
        self.angle_threshold_deg=float(angle_threshold_deg)
        self.gate_last={k:int(np.where((self.plan.cycle_ids==k)&(np.array(self.plan.phases)=="ANGLE_GATE"))[0][-1]) for k in range(cycles)} if angle_gate else {}
        self.cycles=cycles
        self.cycle_ends=np.array([np.where((self.plan.cycle_ids==k)&(np.array(self.plan.phases)=="HOLD_END"))[0][-1] for k in range(cycles)])
        self.max_steps=int(np.ceil(self.plan.times_s[-1]/.014))+cycles*250+5
        self.model = self.plan.model
        m = self.model
        if pip_guard or curl_release:
            main_acts=[m.actuator(f'left_finger{f}_joint{j}_actuator').id for f in (2,3,4) for j in range(1,5)]
            # Better tracking for coordinated wrist motion, retaining the
            # original finite torque limits and all physical contacts.
            m.actuator_gainprm[main_acts,0]*=2.5
            m.actuator_biasprm[main_acts,1]*=2.5
            m.actuator_biasprm[main_acts,2]*=np.sqrt(2.5)
        self.data = mujoco.MjData(m)
        if continuous_knife:
            self.knife_tracking_data=mujoco.MjData(m)
            self.knife_tracking_sim=SimulationModel(m,self.knife_tracking_data,SimulationModel._arm_indices(m,LEFT_ARM),SimulationModel._arm_indices(m,RIGHT_ARM),SimulationModel._hand_indices(m))
            self.knife_tracking_ik=Kinematics(self.knife_tracking_sim,self.knife_tracking_sim.right,'right_tool_tip_site')
        self.qids = m.jnt_qposadr[m.actuator_trnid[:,0]]
        self.dids = m.jnt_dofadr[m.actuator_trnid[:,0]]
        self.pbody = m.body('potato').id
        self.pgeom = m.geom('potato_collision').id
        self.pq = int(m.jnt_qposadr[m.joint('potato_free').id])
        self.pv = int(m.jnt_dofadr[m.joint('potato_free').id])
        self.blade = m.geom('right_knife_blade').id
        self.pads = [m.geom(f'left_finger{f}_pad').id for f in range(1,6)]
        self.hand_geoms = [g for g in range(m.ngeom) if m.geom(g).name.startswith('left_finger')
                           and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.disturbance, self.tactile_enabled = disturbance, tactile_enabled
        self.action_space = spaces.Box(-1,1,(4,),np.float32)
        self.observation_space = spaces.Dict({
            'proprio': spaces.Box(-np.inf,np.inf,(2*m.nu,),np.float32),
            'tactile': spaces.Box(-np.inf,np.inf,(80,),np.float32),
            'privileged': spaces.Box(-np.inf,np.inf,(16,),np.float32),
            'phase': spaces.Box(0,1,(5,),np.float32)})
        self._directions = self._pressure_directions()
        self.done = True

    def _pressure_directions(self):
        m, d = self.model, mujoco.MjData(self.model)
        directions = np.zeros((len(self.plan.qpos),m.nu,3))
        for i, pose in enumerate(self.plan.qpos):
            d.qpos[:] = pose
            mujoco.mj_forward(m,d)
            for column,f in enumerate((1,2,3)):
                pad = self.pads[f]
                jac = np.zeros((3,m.nv)); mujoco.mj_jacGeom(m,d,jac,None,pad)
                acts = [m.actuator(f'left_finger{f+1}_joint{j}_actuator').id for j in range(1,5)]
                j = jac[:,self.dids[acts]]
                toward = np.array([0., 0., -1.])
                delta = j.T@np.linalg.solve(j@j.T+1e-6*np.eye(3),toward*.003)
                directions[i,acts,column] = delta*min(1,.12/max(1e-9,np.max(np.abs(delta))))
        return directions

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m,d = self.model,self.data
        mujoco.mj_resetData(m,d)
        d.qpos[:] = self.plan.qpos[0]
        d.ctrl[:] = d.qpos[self.qids]
        for _ in range(500): mujoco.mj_step(m,d)
        self.initial = d.qpos[self.pq:self.pq+7].copy()
        d.time = 0.
        self.clock=0.; self.steps=0; self.previous=np.zeros(4);self.done=False
        angle = self.np_random.uniform(-np.pi,np.pi)
        self.push = np.array([np.cos(angle),np.sin(angle),0.])*2
        self.push_start = float(self.plan.times_s[np.where(np.array(self.plan.phases)=='TRIO_RETREAT')[0][0]])
        self.max_drift = 0.
        self.cycle_checks = []
        self.gate_ready_s=0.
        self.gate_wait_s=0.
        self.gate_events=[]
        self.gate_released=set()
        self.retract_offsets={}
        self.wrist_holds={}
        self.wrist_end_holds={}
        self.support_bridge_starts={}
        self.retargeted_references={}
        self.extension_limits=None
        self.push_onset = None
        self.applied_impulse = np.zeros(3)
        info=self.measure()
        return self.observation(),info

    def measure(self):
        m,d=self.model,self.data
        helper_loads=np.zeros(2)
        tip_forces=np.zeros((5,3))
        loads=np.zeros(5); slips=np.zeros(5); knife_force=0.; penetration=0.
        for i,c in enumerate(d.contact):
            if c.efc_address < 0:continue
            pair=(int(c.geom1),int(c.geom2));w=np.zeros(6)
            mujoco.mj_contactForce(m,d,i,w)
            if self.blade in pair and any(g in self.hand_geoms for g in pair):knife_force+=max(0,w[0])
            if self.pgeom not in pair:continue
            penetration=max(penetration,-float(c.dist))
            other = pair[1] if pair[0] == self.pgeom else pair[0]
            for k, finger in enumerate((1, 5)):
                if m.geom(other).name.startswith(f'left_finger{finger}_'):
                    helper_loads[k] += max(0,w[0])
            for f,pad in enumerate(self.pads):
                if pad not in pair:continue
                loads[f]+=max(0,w[0])
                tip_forces[f]+=(1 if pair[1]==self.pgeom else -1)*c.frame.reshape(3,3).T@w[:3]
                velocities=[]
                for body in (int(m.geom_bodyid[pad]),self.pbody):
                    jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,c.pos,body)
                    velocities.append(jac@d.qvel)
                relative=velocities[0]-velocities[1]; normal=c.frame[:3]
                slips[f]=max(slips[f],np.linalg.norm(relative-normal*np.dot(normal,relative)))
        drift=float(np.linalg.norm(d.qpos[self.pq:self.pq+3]-self.initial[:3]))
        self.max_drift=max(self.max_drift,drift)
        angle=2*np.arccos(np.clip(abs(np.dot(d.qpos[self.pq+3:self.pq+7],self.initial[3:])),0,1))
        angles=[]
        for f in (2,3,4):
            v=d.xanchor[m.joint(f'left_finger{f}_joint4').id]-d.xanchor[m.joint(f'left_finger{f}_joint3').id]
            angles.append(float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2])))))
        frame=min(int(self.clock/.02),len(self.plan.phases)-1)
        cycle=int(self.plan.cycle_ids[frame])
        completed_support=int(np.sum(frame>=self.support_ends[cycle])) if self.support_repeats>1 else int(cycle in self.gate_released)
        pip_geoms=[m.geom(f'left_finger{f}_link3_collision').id for f in (2,3,4)]
        pip_gap=min(mujoco.mj_geomDistance(m,d,self.blade,g,.1,None) for g in pip_geoms)
        return dict(knife_pip_gap_m=float(pip_gap),regrasp_speed=self.regrasp_speed,support_repeats=self.support_repeats,support_repeats_completed=completed_support,
                    main_tip_positions_m=[d.geom_xpos[self.pads[f]].tolist() for f in (1,2,3)],
                    pip_positions_m=[d.xanchor[m.joint(f"left_finger{f}_joint3").id].tolist() for f in (2,3,4)],
                    middle_angles_deg=angles,angle_gate_threshold_deg=self.angle_threshold_deg,
                    angle_gate_events=list(self.gate_events),cycle=int(self.plan.cycle_ids[min(int(self.clock/.02),len(self.plan.phases)-1)])+1,
                    total_cycles=self.cycles,completed_cycles=int(np.sum(self.clock>=self.plan.times_s[self.cycle_ends])),
                    phase=self.plan.phases[min(int(self.clock/.02),len(self.plan.phases)-1)],
                    displacement_m=drift,max_displacement_m=self.max_drift,rotation_deg=float(np.degrees(angle)),
                    tip_forces_on_potato_n=tip_forces.tolist(),helper_loads_n=helper_loads.tolist(),tip_loads_n=loads.tolist(),tip_slip_m_s=slips.tolist(),knife_hand_force_n=float(knife_force),
                    knife_hand_gap_m=float(min(mujoco.mj_geomDistance(m,d,self.blade,g,.1,None) for g in self.hand_geoms)),
                    knife_potato_vertical_clearance_m=knife_food_vertical_clearance(m,d,self.blade,self.pgeom),
                    penetration_m=penetration, applied_impulse_ns=self.applied_impulse.tolist())

    def middle_angles(self):
        result=[]
        for f in (2,3,4):
            v=self.data.xanchor[self.model.joint(f'left_finger{f}_joint4').id]-self.data.xanchor[self.model.joint(f'left_finger{f}_joint3').id]
            result.append(float(np.degrees(np.arctan2(abs(v[2]),np.linalg.norm(v[:2])))))
        return result

    def observation(self):
        d=self.data
        tactile=hand_tactile(self.model,d).ravel()
        if not self.tactile_enabled:tactile[:]=0
        return dict(proprio=np.r_[d.qpos[self.qids],d.qvel[self.dids]].astype(np.float32),tactile=tactile,
                    privileged=np.r_[[.04,.065,.03],d.qpos[self.pq:self.pq+3]-self.initial[:3],
                        d.qpos[self.pq+3:self.pq+7],d.qvel[self.pv:self.pv+6]].astype(np.float32),
                    phase=np.array([self.clock/self.plan.times_s[-1],*np.array(self.middle_angles())/90.,min(self.gate_ready_s/.06,1.)],np.float32))

    def _retarget_fingers_for_held_wrist(self, cycle, held):
        """Keep fingertip world targets when an early gate stops the wrist."""
        scratch=mujoco.MjData(self.model)
        solvers=[FingerSolver(self.model,scratch,f) for f in (2,3,4)]
        sim=SimulationModel(self.model,scratch,SimulationModel._arm_indices(self.model,LEFT_ARM),
            SimulationModel._arm_indices(self.model,RIGHT_ARM),SimulationModel._hand_indices(self.model))
        knife_ik=Kinematics(sim,sim.right,'right_tool_tip_site')
        wrist_ik=Kinematics(sim,sim.left,'left_palm_tcp_site')
        wrist_base=wrist_ik.fk(held)
        landing=np.where((self.plan.cycle_ids==cycle)&(np.array(self.plan.phases)=='TRIO_PLACE'))[0]
        arm=np.flatnonzero(np.isin(self.qids,self.plan.left_qpos_ids))
        indices=np.where((self.plan.cycle_ids==cycle)&np.isin(self.plan.phases,
            ['KNUCKLE_BACK','KNIFE_CLEAR','PRE_LIFT_HOLD','TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE','HOLD_END']))[0]
        for index in indices:
            scratch.qpos[:]=self.plan.qpos[index]
            mujoco.mj_fwdPosition(self.model,scratch)
            targets=[scratch.geom_xpos[f.tip].copy() for f in solvers]
            dips=[scratch.qpos[f.qids[3]] for f in solvers]
            wrist=held
            if self.landing_wrist_retreat_m:
                u=np.clip((index-landing[0])/max(1,landing[-1]-landing[0]),0,1)
                alpha=u*u*u*(10-15*u+6*u*u)
                pose=wrist_base.copy();pose[1,3]+=alpha*self.landing_wrist_retreat_m
                solution=wrist_ik.ik(pose,held,tolerance=1e-6)
                if not solution.success:raise ValueError('Landing wrist retreat unreachable')
                wrist=solution.joints_rad
            scratch.qpos[self.plan.left_qpos_ids]=wrist
            for f,target,dip in zip(solvers,targets,dips):
                try:
                    f.solve(target,dip_angle=dip)
                except ValueError as exc:
                    raise ValueError(f"Retarget cycle {cycle+1}, frame {index}, {self.plan.phases[index]}, finger {f.finger}: {exc}") from exc
            mujoco.mj_fwdPosition(self.model,scratch)
            for _ in range(8):
                gap=min(mujoco.mj_geomDistance(self.model,scratch,self.blade,g,.1,None) for g in self.hand_geoms)
                close_enough=abs(gap-self.knife_gap_m)<.0001 if self.continuous_knife else gap>=self.knife_gap_m-.0001
                if close_enough:break
                joints=scratch.qpos[sim.right.qpos_ids].copy()
                target=knife_ik.fk(joints);target[1,3]+=gap-self.knife_gap_m
                result=knife_ik.ik(target,joints,tolerance=1e-6)
                if not result.success:raise ValueError('Retargeted knife clearance unreachable')
                scratch.qpos[sim.right.qpos_ids]=result.joints_rad
                mujoco.mj_fwdPosition(self.model,scratch)
            else:raise ValueError('Retargeted knife clearance failed')
            self.retargeted_references[index]=scratch.qpos[self.qids].copy()
        self.wrist_end_holds[cycle]=wrist.copy()

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(abs(action)>1):raise ValueError('Expected four bounded finite actions')
        if self.done:raise RuntimeError('reset required')
        m,d=self.model,self.data
        index=min(int(self.clock/.02),len(self.plan.qpos)-1)
        reference=self.retargeted_references.get(index,self.plan.qpos[index,self.qids]).copy()
        cycle=int(self.plan.cycle_ids[index])
        if cycle in self.retract_offsets and self.plan.phases[index]=='KNUCKLE_BACK':
            start,end,offset=self.retract_offsets[cycle]
            reference+=offset*np.clip((end-self.clock)/(end-start),0,1)
        if self.wrist_retreat_m:
            arm = np.flatnonzero(np.isin(self.qids, self.plan.left_qpos_ids))
            if cycle in self.wrist_holds:
                if not self.landing_wrist_retreat_m or index not in self.retargeted_references:
                    reference[arm] = self.wrist_holds[cycle]
            elif cycle-1 in self.wrist_holds and self.plan.phases[index]=='WRIST_FOLLOW':
                ids=np.where((self.plan.cycle_ids==cycle)&(np.array(self.plan.phases)=='WRIST_FOLLOW'))[0]
                alpha=(index-ids[0]+1)/len(ids)
                reference[arm]=(1-alpha)*self.wrist_end_holds.get(cycle-1,self.wrist_holds[cycle-1])+alpha*self.plan.qpos[ids[-1],self.qids[arm]]
        pressure_scale=1.0
        if self.support_repeats>1 and self.plan.phases[index]=='WRIST_FOLLOW':
            # Start from the commanded end of the previous physical round,
            # not the unretargeted offline hand pose (which causes tip slip).
            first=self.support_bridge_starts.setdefault(cycle,d.ctrl.copy())
            ids=np.where((self.plan.cycle_ids==cycle)&(np.array(self.plan.phases)=='WRIST_FOLLOW'))[0]
            u=(index-ids[0]+1)/len(ids)
            alpha=u*u*u*(10-15*u+6*u*u)
            reference=(1-alpha)*first+alpha*self.plan.qpos[ids[-1],self.qids]
            pressure_scale=alpha
        command=np.clip(reference+pressure_scale*self._directions[index]@action[:3],
                        m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
        if self.curl_release:
            protected=('TRIO_PLACE','HOLD_END') if self.support_repeats>1 else ('TRIO_PLACE','HOLD_END','WRIST_FOLLOW','KNIFE_APPROACH','HOLD')
            ids=[m.actuator(f'left_finger{f}_joint{j}_actuator').id for f in (2,3,4) for j in (3,4)]
            if self.plan.phases[index]=='TRIO_PLACE' and self.extension_limits is None:
                self.extension_limits=d.ctrl[ids].copy()
            if self.plan.phases[index] in protected and self.extension_limits is not None:
                command[ids]=np.minimum(np.minimum(command[ids],self.extension_limits),d.qpos[self.qids[ids]])
                self.extension_limits=command[ids].copy()
            else:self.extension_limits=None
        if self.continuous_knife and self.plan.phases[index]!='INITIAL_APPROACH':
            scratch=self.knife_tracking_data;sim=self.knife_tracking_sim;ik=self.knife_tracking_ik
            scratch.qpos[:]=d.qpos;scratch.qpos[self.qids]=command
            right=sim.right.qpos_ids
            joints=scratch.qpos[right].copy();target=ik.fk(joints)
            for _ in range(12):
                mujoco.mj_fwdPosition(m,scratch)
                gap=min(mujoco.mj_geomDistance(m,scratch,self.blade,g,.1,None) for g in self.hand_geoms)
                if abs(gap-self.knife_gap_m)<.0001:break
                target[1,3]+=gap-self.knife_gap_m
                solved=ik.ik(target,joints,tolerance=1e-6)
                if not solved.success:raise ValueError('Continuous knife tracking unreachable')
                joints=solved.joints_rad;scratch.qpos[right]=joints
            else:raise ValueError('Continuous knife tracking failed')
            acts=sim.right.actuator_ids
            command[acts]=np.clip(joints,m.actuator_ctrlrange[acts,0],m.actuator_ctrlrange[acts,1])
        d.ctrl[:]=command
        maxknife=0.;maxpenetration=0.;maxhelper=0.
        if self.disturbance and self.push_onset is None and self.clock >= self.push_start:
            self.push_onset = float(d.time)
        for _ in range(10):
            d.xfrc_applied[self.pbody]=0
            if self.push_onset is not None and d.time-self.push_onset < .08-1e-9:
                d.xfrc_applied[self.pbody,:3]=self.push
                self.applied_impulse += self.push*m.opt.timestep
            mujoco.mj_step(m,d)
            # Capture transient contact failures at the physics rate.
            for ci,c in enumerate(d.contact):
                pair=(int(c.geom1),int(c.geom2))
                if self.pgeom in pair:
                    maxpenetration=max(maxpenetration,-float(c.dist))
                    other=pair[1] if pair[0]==self.pgeom else pair[0]
                    if any(m.geom(other).name.startswith(f'left_finger{f}_') for f in (1,5)):
                        wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench)
                        maxhelper=max(maxhelper,float(wrench[0]))
                if self.blade in pair and any(g in self.hand_geoms for g in pair):
                    wrench=np.zeros(6);mujoco.mj_contactForce(m,d,ci,wrench);maxknife=max(maxknife,float(wrench[0]))
        d.xfrc_applied[self.pbody]=0
        previous_clock=self.clock
        phase_speed=self.regrasp_speed if self.plan.phases[index] in ('TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE') else 1.0
        next_clock=min(self.plan.times_s[-1],self.clock+.02*phase_speed*(1+.3*action[-1]))
        cycle=int(self.plan.cycle_ids[index])
        if self.angle_gate_enabled and cycle not in self.gate_released:
            gate_time=self.plan.times_s[self.gate_last[cycle]]
            if self.plan.phases[index] in ('CUT_ADVANCE','ANGLE_GATE'):
                gate_info=self.measure()
                repetitions_done=self.support_repeats==1 or index>=self.support_ends[cycle][-1]
                ready=repetitions_done and min(gate_info['middle_angles_deg'])>=self.angle_threshold_deg and min(gate_info['tip_loads_n'][1:4])>.1
                self.gate_ready_s=self.gate_ready_s+.02 if ready else 0.
                if self.gate_ready_s>=.06-1e-9:
                    self.gate_released.add(cycle)
                    if self.wrist_retreat_m:
                        arm=np.flatnonzero(np.isin(self.qids,self.plan.left_qpos_ids))
                        self.wrist_holds[cycle]=reference[arm].copy()
                    self.gate_events.append(dict(cycle=cycle+1,physical_time_s=float(d.time),
                        angles_deg=gate_info['middle_angles_deg'],main_tip_loads_n=gate_info['tip_loads_n'][1:4]))
                    retract=np.where((self.plan.cycle_ids==cycle)&(np.array(self.plan.phases)=='KNUCKLE_BACK'))[0]
                    if self.curl_release:
                        self._retarget_fingers_for_held_wrist(cycle,self.wrist_holds[cycle])
                    start=self.plan.times_s[retract[0]];end=self.plan.times_s[retract[-1]]
                    self.retract_offsets[cycle]=(start,end,reference-self.retargeted_references.get(int(retract[0]),self.plan.qpos[retract[0],self.qids]))
                    next_clock=start
                    self.gate_ready_s=0.;self.gate_wait_s=0.
                elif next_clock>=gate_time:
                    self.gate_wait_s+=.02
                    next_clock=gate_time
        self.clock=next_clock;self.steps+=1
        info=self.measure();info['knife_hand_force_n']=max(maxknife,info['knife_hand_force_n']);info['penetration_m']=max(maxpenetration,info['penetration_m'])
        info['max_helper_contact_force_n']=maxhelper
        loads=np.asarray(info['tip_loads_n']);slip=np.asarray(info['tip_slip_m_s'])
        phase=info['phase']
        required=np.array([], dtype=int) if phase in ('TRIO_LIFT','TRIO_RETREAT') else np.array([1,2,3])
        reward=float((self.clock-previous_clock)/.02-60*info['displacement_m']-4*np.linalg.norm(d.qvel[self.pv:self.pv+3])
                     -3*np.sum(slip)-.2*np.linalg.norm(np.sum(info["tip_forces_on_potato_n"],axis=0)[:2])-.02*info['rotation_deg']+.1*np.minimum(loads[required],1).sum()
                     -.03*np.maximum(loads-3,0).sum()-.02*np.sum((action-self.previous)**2))
        while len(self.cycle_checks) < info['completed_cycles']:
            verified=bool(self.max_drift<.005 and np.all(loads[1:4]>.1)
                          and np.linalg.norm(d.qvel[self.pv:self.pv+3])<.02)
            self.cycle_checks.append(dict(cycle=len(self.cycle_checks)+1,verified=verified,
                max_displacement_m=self.max_drift,displacement_m=info['displacement_m'],
                main_tip_loads_n=loads[1:4].tolist()))
            reward += 5 if verified else -5
        info['cycle_checks']=list(self.cycle_checks)
        finite=np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        failed=self.gate_wait_s>5.0 or not finite or info['displacement_m']>.04 or info['penetration_m']>.004 or info['knife_hand_force_n']>.5
        complete=self.clock>=self.plan.times_s[-1]
        truncated=self.steps>=self.max_steps and not (failed or complete)
        self.done=bool(failed or complete or truncated); self.previous=action.copy()
        info['is_success']=bool(complete and not failed and self.max_drift<.005 and len(self.cycle_checks)==self.cycles
                                and all(c['verified'] for c in self.cycle_checks))
        info['termination_reason']='angle_gate_timeout' if self.gate_wait_s>5.0 else 'failure' if failed else 'complete' if complete else 'timeout' if truncated else 'running'
        if failed:reward-=30
        return self.observation(),reward,bool(failed or complete),bool(truncated),info
