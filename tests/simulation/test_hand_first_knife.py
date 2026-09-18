"""Verify physical left-hand completion precedes right-hand advancement."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_knife_waits_for_reseated_hand_then_advances_smoothly(monkeypatch):
    env=DynamicRegraspEnv(cycles=1,wrist_retreat_m=.004,smooth_regrasp=True,
        angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004,
        early_pip_curl=True,support_repeats=3,knife_gap_m=.007,cut_depth_m=.024,
        regrasp_speed=1.6,continuous_knife=True,no_knife_reverse=True,hand_first_knife=True)
    env.reset(seed=100)
    held=[];actual=[];following=[];last_place_time=0.;blocked=False
    for _ in range(env.max_steps):
        phase=env.plan.phases[min(int(env.clock/.02),len(env.plan.phases)-1)]
        if phase=='HOLD_END' and not blocked:
            # Simulate missing tactile confirmation after the trajectory ends:
            # elapsed reference time alone must never release the knife.
            measure=env.measure
            def no_touch():
                info=measure();info['tip_loads_n']=[0.]*5
                return info
            held_y=env.knife_last_y
            with monkeypatch.context() as patch:
                patch.setattr(env,'measure',no_touch)
                for _ in range(5):
                    _,_,done,truncated,_=env.step(np.zeros(4))
                    assert not done and not truncated
                    assert not env.knife_follow_released
                    assert abs(env.knife_last_y-held_y)<1e-8
            blocked=True
        released=0 in env.knife_follow_released
        _,_,done,truncated,info=env.step(np.zeros(4))
        if phase in ('TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE') or (phase=='HOLD_END' and not released):
            held.append(env.knife_last_y)
            actual.append(info['knife_position_m'][1])
        if phase=='TRIO_PLACE':last_place_time=env.data.time
        if phase=='HOLD_END' and released:following.append(env.knife_last_y)
        assert info['knife_hand_force_n']==0
        assert info['knife_hand_gap_m']>0
        assert info['knife_potato_vertical_clearance_m']>0
        if done or truncated:break
    assert info['termination_reason']=='complete'
    assert len(held)>20 and np.ptp(held)<1e-8
    assert np.ptp(actual)<.0003
    assert len(following)>10 and following[-1]-held[-1]>.002
    assert np.min(np.diff(following))>=-1e-8
    assert np.max(np.diff(following))<=.020*.02+1e-8
    event=info['knife_follow_events'][0]
    assert event['hand_ready_time_s']>=last_place_time+.06-1e-8
    assert event['knife_caught_time_s']>event['hand_ready_time_s']+.2
    assert min(event['main_tip_loads_n'])>.1
    assert max(event['main_tip_slip_m_s'])<.02
    assert info['knife_follow_done']
    env.reset(seed=100)
    assert not env.knife_follow_events and not env.knife_follow_released
