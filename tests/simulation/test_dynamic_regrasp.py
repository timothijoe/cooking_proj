"""Dynamics checks independent of the geometric preview's fixed object poses."""
import mujoco
import numpy as np
import pytest
from twin_sim.dynamic_regrasp import DynamicRegraspEnv

@pytest.fixture(scope='module')
def env():
    return DynamicRegraspEnv(cycles=1)

def test_actions_are_finite_bounded_and_reset_is_required(env):
    env.reset(seed=100)
    for action in (np.zeros(3),np.full(4,np.nan),np.full(4,1.01)):
        with pytest.raises(ValueError):env.step(action)
    assert env.observation_space.contains(env.observation())

def test_potato_velocity_is_integrated_not_overwritten(env):
    env.reset(seed=100)
    env.data.qvel[env.pv]=.3
    before=env.data.qpos[env.pq:env.pq+3].copy()
    env.step(np.zeros(4))
    assert np.linalg.norm(env.data.qpos[env.pq:env.pq+3]-before)>.0002

def test_close_knife_and_finite_force_full_cycle(env):
    env.disturbance=False
    env.reset(seed=100)
    metrics=[]
    for _ in range(650):
        _,reward,done,truncated,info=env.step(np.zeros(4));metrics.append(info)
        assert np.isfinite(reward)
        if done or truncated:break
    assert info['termination_reason']=='complete'
    assert max(x['knife_hand_force_n'] for x in metrics)<.01
    assert max(x['max_helper_contact_force_n'] for x in metrics)==0
    assert max(max(x['helper_loads_n']) for x in metrics)==0
    cutting=[x for x in metrics if x['phase']=='KNUCKLE_BACK']
    assert min(x['knife_hand_gap_m'] for x in cutting)>0
    assert max(x['knife_hand_gap_m'] for x in cutting)<.008
    assert min(x['knife_potato_vertical_clearance_m'] for x in metrics)>0
    # The revised contact placement must hold drift below 5 mm without fixing the food.
    assert .0001<info['max_displacement_m']<.005
    assert info['is_success']
    assert any(max(x['tip_loads_n'])>.1 for x in metrics)
    env.disturbance=True

def test_pressure_residuals_change_actuator_targets(env):
    env.reset(seed=100);env.step(np.zeros(4));baseline=env.data.ctrl.copy()
    env.reset(seed=100);env.step(np.r_[np.ones(3),0.])
    assert np.max(abs(env.data.ctrl-baseline))>.001
    assert np.all(env.data.ctrl<=env.model.actuator_ctrlrange[:,1])
    assert np.all(env.data.ctrl>=env.model.actuator_ctrlrange[:,0])

@pytest.mark.parametrize('speed', [-1., 1.])
def test_disturbance_impulse_is_independent_of_reference_speed(env, speed):
    env.disturbance=True
    env.reset(seed=100)
    env.clock=env.push_start
    for _ in range(6):
        _,_,done,truncated,info=env.step(np.r_[np.zeros(3),speed])
        if done or truncated:break
    assert np.linalg.norm(info['applied_impulse_ns'])==pytest.approx(.16,abs=1e-9)


def test_policy_cannot_command_parked_helpers(env):
    env.reset(seed=100)
    m=env.model
    acts=[m.actuator(f'left_finger{f}_joint{j}_actuator').id for f in (1,5) for j in range(1,5)]
    qids=env.qids[acts]
    np.testing.assert_allclose(env.plan.qpos[:,qids],np.broadcast_to(env.plan.qpos[0,qids],(len(env.plan.qpos),8)))
    for direction in (-1.,1.):
        env.reset(seed=100)
        env.step(np.full(4,direction))
        np.testing.assert_allclose(env.data.ctrl[acts],env.plan.qpos[0,qids],atol=1e-12)


def test_default_environment_has_no_external_push():
    plant=DynamicRegraspEnv(cycles=1)
    assert not plant.disturbance
    plant.reset(seed=100)
    for _ in range(650):
        _,_,done,truncated,info=plant.step(np.zeros(4))
        np.testing.assert_array_equal(plant.data.xfrc_applied,np.zeros_like(plant.data.xfrc_applied))
        np.testing.assert_array_equal(info['applied_impulse_ns'],np.zeros(3))
        if done or truncated:break
    assert plant.push_onset is None
    assert info['termination_reason']=='complete'
