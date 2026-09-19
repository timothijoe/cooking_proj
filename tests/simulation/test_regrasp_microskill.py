"""Physical and command-contract checks for the independent reference skill."""
import mujoco
import numpy as np
import pytest

from twin_sim.regrasp_microskill import MicroSkillConfig, RegraspMicroSkill


def finish(skill):
    for _ in range(1200):
        skill.step()
        if skill.phase in ('HOLD', 'FAULT'):
            return
    pytest.fail('skill did not terminate')


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_suspended_start_and_three_physical_requests(seed):
    skill = RegraspMicroSkill(MicroSkillConfig(seed=seed, randomize_shape=seed > 0))
    assert skill.model.nu == 27
    assert mujoco.mj_name2id(skill.model, mujoco.mjtObj.mjOBJ_BODY, 'right_link1') == -1
    assert np.all(skill.loads() == 0)
    initial_potato = skill.data.qpos[skill.pq:skill.pq+7].copy()
    times = []
    for request in range(3):
        before = skill.data.qpos.copy()
        assert skill.request(str(request)) == 'ACCEPTED'
        np.testing.assert_array_equal(skill.data.qpos, before)
        assert skill.request(str(request)) == 'DUPLICATE'
        assert skill.request(f'busy-{request}') == 'BUSY'
        finish(skill)
        assert skill.phase == 'HOLD', skill.status()
        assert skill.completed == request+1
        assert skill.reason == 'succeeded'
        times.append(skill.data.time)
        assert np.all(skill.loads() > .1)
        assert skill.max_drift < .005
    assert np.all(np.diff(times) > 0)
    assert not np.array_equal(skill.data.qpos[skill.pq:skill.pq+7], initial_potato)
    assert skill.request('fourth') == 'WORKSPACE_LIMIT'
    clock = skill.data.time
    for _ in range(50):
        skill.step()
    assert skill.phase == 'HOLD'
    assert skill.data.time > clock+.99
    assert np.all(skill.loads() > .1)
    assert all(np.isfinite(v).all() for v in skill.observation().values())


def test_missing_contact_never_advances_and_physics_keeps_running():
    skill = RegraspMicroSkill()
    # A failed tactile channel cannot falsely unlock the motion sequence.
    skill.loads = lambda: np.zeros(3)
    skill.request('no-contact')
    finish(skill)
    assert skill.phase == 'FAULT'
    assert skill.completed == 0
    assert not any(e['phase'] == 'SUPPORT_RETREAT' for e in skill.events)
    clock = skill.data.time
    skill.step()
    assert skill.data.time > clock


def test_cancel_during_descent_holds_and_does_not_claim_support():
    skill = RegraspMicroSkill()
    skill.request('start')
    for _ in range(10):
        skill.step()
    skill.cancel()
    assert skill.phase == 'HOLD'
    assert skill.completed == 0
    assert skill.first_support is None
    control = skill.data.ctrl.copy()
    skill.step()
    np.testing.assert_array_equal(skill.data.ctrl, control)
    assert skill.request('resume') == 'ACCEPTED'
    assert skill.phase == 'APPROACH'


def test_three_wrist_advances_before_each_fingertip_relocation():
    skill = RegraspMicroSkill(MicroSkillConfig(seed=100, randomize_shape=True, support_repeats=3))
    for request in range(3):
        skill.request(str(request))
        positions = []
        for _ in range(1200):
            skill.step()
            if skill.phase == 'SUPPORT_RETREAT':
                positions.append(skill.data.site_xpos[skill.model.site('left_palm_tcp_site').id, 1])
            if skill.phase in ('HOLD', 'FAULT'):
                break
        assert skill.phase == 'HOLD', skill.status()
        assert skill.wrist_retreats == (request+1)*3
        assert positions[-1]-positions[0] > .003
    counts = [e['count'] for e in skill.events if e['phase'] == 'WRIST_RETREAT_DONE']
    assert counts == list(range(1, 10))
    releases = [i for i, e in enumerate(skill.events) if e['phase'] == 'RELEASE_MOVE']
    for cycle, index in enumerate(releases, start=1):
        preceding = [e for e in skill.events[:index] if e['phase'] == 'WRIST_RETREAT_DONE']
        assert preceding[-1]['count'] == cycle*3
    assert skill.completed == 3
