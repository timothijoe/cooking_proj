"""The knife must not back away when the synchronized fingertips regrasp."""
import mujoco
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_regrasp_preserves_knife_direction_and_physical_clearance():
    env = DynamicRegraspEnv(
        cycles=1, wrist_retreat_m=.004, smooth_regrasp=True,
        angle_threshold_deg=80, curl_release=True, landing_wrist_retreat_m=.004,
        early_pip_curl=True, support_repeats=3, knife_gap_m=.007,
        cut_depth_m=.024, regrasp_speed=1.6, continuous_knife=True,
        no_knife_reverse=True,
    )
    env.reset(seed=100)
    commanded = []
    actual = []
    gaps = []
    for _ in range(env.max_steps):
        phase = env.plan.phases[min(int(env.clock/.02), len(env.plan.phases)-1)]
        _, _, done, truncated, info = env.step(np.zeros(4))
        if phase in ('TRIO_LIFT', 'TRIO_RETREAT', 'TRIO_PLACE'):
            commanded.append(env.knife_last_y)
            mujoco.mj_fwdPosition(env.model, env.data)
            actual.append(env.data.geom_xpos[env.blade, 1])
        if phase != 'INITIAL_APPROACH':
            gaps.append(info['knife_hand_gap_m'])
            assert info['knife_hand_force_n'] == 0
            assert info['knife_potato_vertical_clearance_m'] > 0
        if done or truncated:
            break
    assert info['termination_reason'] == 'complete'
    assert len(actual) > 20
    assert np.min(np.diff(commanded)) >= -1e-8
    assert np.max(np.maximum.accumulate(actual)-actual) < .0003
    assert min(gaps) > 0
    assert max(gaps) < .008
    env.reset(seed=100)
    assert env.knife_last_y is None
