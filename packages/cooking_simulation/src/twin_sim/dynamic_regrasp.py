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


def add_close_knife_reference(plan):
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
    active = np.where(np.array(plan.phases) == 'KNUCKLE_BACK')[0]
    previous = d.qpos[sim.right.qpos_ids].copy()
    last_guided = initial.copy()
    for i, phase in enumerate(plan.phases):
        d.qpos[:] = plan.qpos[i]
        mujoco.mj_forward(m, d)
        target = initial.copy()
        # Lower blade so its flat face overlaps the guiding knuckles; make two
        # shallow down/up strokes, stopping above the uncut rigid potato.
        fraction = np.clip((i-active[0])/max(1,len(active)-1), 0, 1)
        stroke = .018*np.sin(2*np.pi*fraction)**2 if phase == 'KNUCKLE_BACK' else 0.
        clear = np.clip((i-active[-1])/30, 0, 1)
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
            if clear == 0 and abs(gap-.003) > .00015:
                target[1,3] += gap-.003
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


def build_continuous_regrasp(cycles=3, contact_shift_m=.024):
    """Sequential slicing references with continuous wrist/knife transitions."""
    if not isinstance(cycles,int) or not 1 <= cycles <= 3:
        raise ValueError('cycles must be an integer from 1 to 3 for the current potato')
    parts=[add_close_knife_reference(park_auxiliary_fingers(build_finger_regrasp_preview(
        bottom_cut_m=.003, fingertip_back_shift_m=contact_shift_m+.008*k,
        palm_lift_m=.016 if k==0 else .017))) for k in range(cycles)]
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
            for label,a,b in [('WRIST_FOLLOW',first,wrist_ready),('KNIFE_APPROACH',wrist_ready,next_pose)]:
                for t in np.linspace(0,1,61)[1:]:
                    alpha=t*t*(3-2*t)
                    states.append((1-alpha)*a+alpha*b);phases.append(label);cycle_ids.append(cycle)
        states.extend(part.qpos.copy());phases.extend(part.phases);cycle_ids.extend([cycle]*len(part.qpos))
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
    def __init__(self, disturbance=False, tactile_enabled=True, contact_shift_m=.024, cycles=3):
        self.plan = build_continuous_regrasp(cycles,contact_shift_m)
        self.cycles=cycles
        self.cycle_ends=np.array([np.where((self.plan.cycle_ids==k)&(np.array(self.plan.phases)=="HOLD_END"))[0][-1] for k in range(cycles)])
        self.max_steps=int(np.ceil(self.plan.times_s[-1]/.014))+5
        self.model = self.plan.model
        m = self.model
        self.data = mujoco.MjData(m)
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
            'phase': spaces.Box(0,1,(1,),np.float32)})
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
        return dict(cycle=int(self.plan.cycle_ids[min(int(self.clock/.02),len(self.plan.phases)-1)])+1,
                    total_cycles=self.cycles,completed_cycles=int(np.sum(self.clock>=self.plan.times_s[self.cycle_ends])),
                    phase=self.plan.phases[min(int(self.clock/.02),len(self.plan.phases)-1)],
                    displacement_m=drift,max_displacement_m=self.max_drift,rotation_deg=float(np.degrees(angle)),
                    tip_forces_on_potato_n=tip_forces.tolist(),helper_loads_n=helper_loads.tolist(),tip_loads_n=loads.tolist(),tip_slip_m_s=slips.tolist(),knife_hand_force_n=float(knife_force),
                    knife_hand_gap_m=float(min(mujoco.mj_geomDistance(m,d,self.blade,g,.1,None) for g in self.hand_geoms)),
                    knife_potato_vertical_clearance_m=knife_food_vertical_clearance(m,d,self.blade,self.pgeom),
                    penetration_m=penetration, applied_impulse_ns=self.applied_impulse.tolist())

    def observation(self):
        d=self.data
        tactile=hand_tactile(self.model,d).ravel()
        if not self.tactile_enabled:tactile[:]=0
        return dict(proprio=np.r_[d.qpos[self.qids],d.qvel[self.dids]].astype(np.float32),tactile=tactile,
                    privileged=np.r_[[.04,.065,.03],d.qpos[self.pq:self.pq+3]-self.initial[:3],
                        d.qpos[self.pq+3:self.pq+7],d.qvel[self.pv:self.pv+6]].astype(np.float32),
                    phase=np.array([self.clock/self.plan.times_s[-1]],np.float32))

    def step(self,action):
        action=np.asarray(action,dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(abs(action)>1):raise ValueError('Expected four bounded finite actions')
        if self.done:raise RuntimeError('reset required')
        m,d=self.model,self.data
        index=min(int(self.clock/.02),len(self.plan.qpos)-1)
        d.ctrl[:]=np.clip(self.plan.qpos[index,self.qids]+self._directions[index]@action[:3],
                          m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])
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
        self.clock=min(self.plan.times_s[-1],self.clock+.02*(1+.3*action[-1]));self.steps+=1
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
        failed=not finite or info['displacement_m']>.04 or info['penetration_m']>.004 or info['knife_hand_force_n']>.5
        complete=self.clock>=self.plan.times_s[-1]
        truncated=self.steps>=self.max_steps and not (failed or complete)
        self.done=bool(failed or complete or truncated); self.previous=action.copy()
        info['is_success']=bool(complete and not failed and self.max_drift<.005 and len(self.cycle_checks)==self.cycles
                                and all(c['verified'] for c in self.cycle_checks))
        info['termination_reason']='failure' if failed else 'complete' if complete else 'timeout' if truncated else 'running'
        if failed:reward-=30
        return self.observation(),reward,bool(failed or complete),bool(truncated),info
