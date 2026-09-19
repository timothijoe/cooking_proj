import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from twin_sim.regrasp_rl import RegraspRLEnv


def test_gym_contract_and_zero_action_completes():
    env = RegraspRLEnv(randomize_shape=False, cycles=1)
    check_env(env, warn=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (354,)
    for _ in range(1000):
        obs, reward, term, trunc, info = env.step(np.zeros(8))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if term or trunc:
            break
    assert info['is_success'], info
    with pytest.raises(RuntimeError):
        env.step(np.zeros(8))


def test_action_changes_physical_command_and_no_privileged_pose_in_observation():
    env = RegraspRLEnv(randomize_shape=False)
    env.reset(seed=0)
    env.step(np.ones(8))
    positive = env.skill.data.ctrl.copy()
    env.reset(seed=0)
    env.step(-np.ones(8))
    negative = env.skill.data.ctrl.copy()
    assert np.max(abs(positive-negative)) > 1e-6
    before = env._frame()
    env.skill.data.qpos[env.skill.pq] += .01
    # Object truth is not read by the actor's frame builder.
    np.testing.assert_array_equal(env._frame(), before)
    with pytest.raises(ValueError):
        env.step(np.full(8, np.nan))
