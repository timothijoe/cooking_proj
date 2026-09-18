"""Landing wrist assistance translates backward without twisting the palm."""
import mujoco
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_landing_wrist_retreat_preserves_orientation_and_tip_targets():
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004)
    env.reset(seed=100)
    for _ in range(env.max_steps):
        _,_,done,trunc,info=env.step(np.zeros(4))
        if done or trunc:break
    assert info['termination_reason']=='complete'
    p=env.plan;d=mujoco.MjData(env.model);site=env.model.site('left_palm_tcp_site').id
    landing=np.where(np.array(p.phases)=='TRIO_PLACE')[0]
    positions=[];rotations=[]
    for i in landing:
        d.qpos[:]=p.qpos[i];mujoco.mj_fwdPosition(env.model,d)
        expected=np.array([d.geom_xpos[env.model.geom(f'left_finger{f}_pad').id].copy() for f in (2,3,4)])
        d.qpos[env.qids]=env.retargeted_references[i];mujoco.mj_fwdPosition(env.model,d)
        actual=np.array([d.geom_xpos[env.model.geom(f'left_finger{f}_pad').id].copy() for f in (2,3,4)])
        np.testing.assert_allclose(actual,expected,atol=.00015)
        positions.append(d.site_xpos[site].copy());rotations.append(d.site_xmat[site].copy())
    positions=np.array(positions);rotations=np.array(rotations)
    np.testing.assert_allclose(positions[-1]-positions[0],[0,.004,0],atol=1e-6)
    assert np.min(np.diff(positions[:,1]))>=-1e-6
    np.testing.assert_allclose(rotations,np.broadcast_to(rotations[0],rotations.shape),atol=1e-5)
