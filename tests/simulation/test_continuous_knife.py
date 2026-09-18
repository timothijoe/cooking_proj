"""Only initialization may move the knife away from the guiding hand."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_knife_stays_close_after_single_initial_approach():
    e=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.004,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004,early_pip_curl=True,support_repeats=3,knife_gap_m=.003,cut_depth_m=.024,regrasp_speed=1.6,continuous_knife=True)
    phases=np.array(e.plan.phases)
    assert list(np.where(phases=='INITIAL_APPROACH')[0])==list(range(50))
    assert 'KNIFE_CLEAR' not in phases and 'KNIFE_APPROACH' not in phases
    e.reset(seed=100)
    initial_gap=e.measure()['knife_hand_gap_m'];gaps=[]
    assert initial_gap>.02
    for _ in range(e.max_steps):
        _,_,done,trunc,info=e.step(np.zeros(4))
        if info['phase']!='INITIAL_APPROACH':
            gaps.append(info['knife_hand_gap_m'])
            assert info['knife_hand_force_n']<.5
            assert info['knife_potato_vertical_clearance_m']>0
        if done or trunc:break
    assert info['termination_reason']=='complete'
    assert min(gaps)>0
    assert max(gaps)<.008
