"""Contact-anchored wrist retreat followed by fingers with a held wrist."""
import mujoco
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_wrist_retreat_then_hold_and_synchronized_fingers():
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.008)
    p=env.plan;d=mujoco.MjData(env.model);site=env.model.site('left_palm_tcp_site').id
    positions=[];rotations=[]
    for q in p.qpos:
        d.qpos[:]=q;mujoco.mj_fwdPosition(env.model,d)
        positions.append(d.site_xpos[site].copy());rotations.append(d.site_xmat[site].copy())
    positions=np.asarray(positions);rotations=np.asarray(rotations)
    np.testing.assert_allclose(rotations,np.broadcast_to(rotations[0],rotations.shape),atol=1e-5)
    cut=np.where(np.array(p.phases)=='CUT_ADVANCE')[0]
    np.testing.assert_allclose(p.tips_m[cut],np.broadcast_to(p.tips_m[cut[0]],p.tips_m[cut].shape),atol=.0001)
    assert np.all(np.diff(positions[cut,1])>=-1e-6)
    assert positions[cut[-1],1]-positions[0,1]>.00799
    after=np.where(np.isin(p.phases,['KNUCKLE_BACK','TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE']))[0]
    np.testing.assert_allclose(positions[after],np.broadcast_to(positions[after[0]],positions[after].shape),atol=1e-6)
    env.reset(seed=100);held=None
    arm=np.flatnonzero(np.isin(env.qids,p.left_qpos_ids))
    for _ in range(env.max_steps):
        _,_,done,trunc,info=env.step(np.zeros(4))
        if held is not None:np.testing.assert_allclose(env.data.ctrl[arm],held,atol=1e-12)
        if env.wrist_holds:held=env.wrist_holds[0].copy()
        if done or trunc:break
    assert held is not None
    assert info['termination_reason']=='complete'
