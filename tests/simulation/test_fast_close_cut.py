"""Speed up actual regrasp dynamics while retaining the support/cutting stage."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_faster_regrasp_preserves_three_supports_and_knife_clearance():
    e=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.004,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004,early_pip_curl=True,support_repeats=3,knife_gap_m=.003,cut_depth_m=.024)
    counts=[]
    for speed in (1.,1.6):
        e.regrasp_speed=speed;e.reset(seed=100)
        moving=0;support=0;min_gap=1.
        for _ in range(e.max_steps):
            _,_,done,trunc,info=e.step(np.zeros(4))
            moving+=info['phase'] in ('TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE')
            support+=info['phase']=='CUT_ADVANCE'
            min_gap=min(min_gap,info['knife_hand_gap_m'])
            assert info['knife_hand_force_n']<.5
            assert info['knife_potato_vertical_clearance_m']>0
            if info['angle_gate_events']:assert info['support_repeats_completed']==3
            if done or trunc:break
        assert info['termination_reason']=='complete'
        assert min_gap>0
        counts.append((moving,support))
    assert counts[1][0]<counts[0][0]*.7
    assert counts[1][1]==counts[0][1]
