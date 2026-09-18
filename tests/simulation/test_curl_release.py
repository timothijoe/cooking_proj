"""Constrain finger articulation, not only the fingertip Cartesian path."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_pip_dip_curl_on_departure_and_extend_on_landing():
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True)
    p=env.plan
    lift=np.where(np.array(p.phases)=='TRIO_LIFT')[0]
    place=np.where(np.array(p.phases)=='TRIO_PLACE')[0]
    for f in (2,3,4):
        ids=[env.model.jnt_qposadr[env.model.joint(f'left_finger{f}_joint{j}').id] for j in (3,4)]
        q=p.qpos[:,ids]
        assert np.min(q[lift[-1]]-q[lift[0]-1])>np.radians(1)
        assert np.max(q[place[-1]]-q[place[0]])<np.radians(-1)
        assert np.max(np.diff(q[place],axis=0))<np.radians(.03)
        assert abs(q[place[-1],1]-np.radians(30))<np.radians(.1)


def test_round_bridge_preserves_open_dip_instead_of_restoring_curled_seed():
    env=DynamicRegraspEnv(cycles=3,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True)
    p=env.plan
    ids=[env.model.jnt_qposadr[env.model.joint(f'left_finger{f}_joint4').id] for f in (2,3,4)]
    phases=np.array(p.phases)
    for cycle in (1,2):
        k=np.where((p.cycle_ids==cycle)&np.isin(phases,['WRIST_FOLLOW','KNIFE_APPROACH','HOLD']))[0]
        q=p.qpos[np.ix_(k,ids)]
        assert np.max(abs(q-np.radians(30)))<np.radians(.15)
        assert np.max(q-q[0])<np.radians(.15)


def test_landing_controller_cannot_recurl_pip_or_dip():
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True)
    env.reset(seed=100)
    ids=[env.model.actuator(f'left_finger{f}_joint{j}_actuator').id for f in (2,3,4) for j in (3,4)]
    last=None
    for _ in range(env.max_steps):
        _,_,done,trunc,info=env.step(np.zeros(4))
        if env.extension_limits is not None:
            if last is not None:assert np.max(env.data.ctrl[ids]-last)<1e-12
            last=env.data.ctrl[ids].copy()
        if done or trunc:break
    assert last is not None
    assert info['termination_reason']=='complete'
