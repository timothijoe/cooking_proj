import importlib
import importlib.util

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="install the optional potato dependencies")
from robot_core.paths import data_root

pytestmark = pytest.mark.skipif(
    not (data_root()/"recordings/recorded_hand_guarded_chop_200hz_latest.npz").is_file(),
    reason="local recorded arm/hand reference is unavailable",
)


def make_env(**kwargs):
    assert importlib.util.find_spec("twin_sim.potato_env") is not None, "potato RL environment is missing"
    return importlib.import_module("twin_sim.potato_env").PotatoRegraspEnv(**kwargs)


def test_reset_is_seeded_and_observation_is_valid():
    with make_env() as env:
        a, _ = env.reset(seed=42)
        b, _ = env.reset(seed=42)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        assert env.observation_space.contains(a)
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "potato_hand_force_n" in info


def test_invalid_action_does_not_advance_physics():
    with make_env() as env:
        env.reset(seed=1)
        time = env.data.time
        with pytest.raises(ValueError, match="action"):
            env.step(np.full(env.action_space.shape, np.nan))
        assert env.data.time == time


def test_step_uses_dynamics_and_does_not_teleport_potato():
    with make_env(disturbance=False) as env:
        env.reset(seed=1)
        env.data.qvel[env.potato_dof:env.potato_dof+3] = [0.1, 0, 0]
        x = env.data.qpos[env.potato_qpos]
        env.step(np.zeros(env.action_space.shape))
        assert env.data.qpos[env.potato_qpos] != x
        assert env.data.time == pytest.approx(.01)


def test_gym_contract():
    from gymnasium.utils.env_checker import check_env
    with make_env() as env:
        check_env(env, skip_render_check=True)


def test_reset_provides_a_settled_contact_state():
    with make_env() as env:
        _, info = env.reset(seed=0)
        assert info["potato_penetration_m"] < .002
        assert not info["knife_hand_contact"]
        assert env.data.time == 0
