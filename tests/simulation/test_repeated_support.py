"""Each contact location supports three wrist retreats before fingers move."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_three_retreats_keep_reference_tips_and_gate_waits_for_all_three():
    e=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.004,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004,early_pip_curl=True,support_repeats=3)
    p=e.plan;cut=np.where(np.array(p.phases)=='CUT_ADVANCE')[0]
    assert len(cut)==180
    np.testing.assert_allclose(p.tips_m[cut],np.broadcast_to(p.tips_m[cut[0]],p.tips_m[cut].shape),atol=.0001)
    arm=p.left_qpos_ids
    for k in range(3):
        hold=cut[k*60+40:k*60+60]
        np.testing.assert_allclose(p.qpos[np.ix_(hold,arm)],np.broadcast_to(p.qpos[hold[0],arm],(len(hold),len(arm))),atol=1e-9)
    e.reset(seed=100)
    for _ in range(e.max_steps):
        _,_,done,trunc,info=e.step(np.zeros(4))
        if info['phase']=='CUT_ADVANCE':
            assert min(info['tip_loads_n'][1:4])>.1
        if info['angle_gate_events']:
            assert info['support_repeats_completed']==3
        if done or trunc:break
    assert info['termination_reason']=='complete'
    assert info['angle_gate_events']
    assert min(info['angle_gate_events'][0]['angles_deg'])>=80
