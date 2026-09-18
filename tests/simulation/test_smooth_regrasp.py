"""The three finger phases label one continuous path without internal stops."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_smooth_arc_and_earlier_contact_gate():
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80)
    p=env.plan
    ids=np.where(np.isin(p.phases,['TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE']))[0]
    tips=p.tips_m[ids]
    dy=np.diff(tips[:,:,1],axis=0)
    assert np.min(dy)>-3e-5
    for i in range(1,len(ids)-1):
        if p.phases[ids[i]]!=p.phases[ids[i-1]]:
            assert np.min(np.linalg.norm(tips[i+1]-tips[i-1],axis=1))>.00015
    assert env.angle_threshold_deg==80
    env.reset(seed=100)
    for _ in range(env.max_steps):
        _,_,done,trunc,info=env.step(np.zeros(4))
        if done or trunc:break
    assert info['angle_gate_events']
    angles=info['angle_gate_events'][0]['angles_deg']
    assert 80 <= min(angles) < 85
    assert info['termination_reason']=='complete'
