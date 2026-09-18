"""Three sequential cutting/regrasp rounds in one physical episode."""
import numpy as np
import pytest
from twin_sim.dynamic_regrasp import DynamicRegraspEnv

@pytest.fixture(scope='module')
def rollout():
    env=DynamicRegraspEnv(cycles=3)
    env.reset(seed=100)
    rows=[];positions=[];times=[]
    for _ in range(env.max_steps):
        _,_,done,truncated,info=env.step(np.zeros(4))
        rows.append(info);positions.append(env.data.qpos[env.pq:env.pq+3].copy());times.append(env.data.time)
        if done or truncated:break
    return env,rows,np.array(positions),np.array(times)


def test_three_rounds_accumulate_retreat_without_reset(rollout):
    env,rows,positions,times=rollout
    assert set(row['cycle'] for row in rows)=={1,2,3}
    assert rows[-1]['completed_cycles']==3
    assert rows[-1]['termination_reason']=='complete'
    assert rows[-1]['is_success']
    assert len(rows[-1]['cycle_checks'])==3
    assert all(c['verified'] for c in rows[-1]['cycle_checks'])
    assert rows[-1]['max_displacement_m']<.005
    assert np.all(np.diff(times)>0)
    assert times[-1]>29
    assert np.max(np.linalg.norm(np.diff(positions,axis=0),axis=1))<.0005
    assert np.all(np.diff([x['max_displacement_m'] for x in rows])>=0)
    p=env.plan
    np.testing.assert_allclose(p.tips_m[-1,:,1]-p.tips_m[0,:,1],.024,atol=.0002)
    assert np.max(abs(np.diff(p.qpos[:,env.qids],axis=0)))<.03


def test_cycle_transitions_keep_helpers_clear_and_knife_outside(rollout):
    env,rows,_,_=rollout
    assert 'WRIST_FOLLOW' in env.plan.phases and 'KNIFE_APPROACH' in env.plan.phases
    assert min(x['knife_hand_gap_m'] for x in rows)>0
    assert min(x['knife_potato_vertical_clearance_m'] for x in rows)>0
    assert max(x['knife_hand_force_n'] for x in rows)==0
    assert max(x['max_helper_contact_force_n'] for x in rows)==0
    assert all(x['applied_impulse_ns']==[0.,0.,0.] for x in rows)
    assert env.plan.finger_clearances_m.min()>.0005


def test_supported_cycle_count_is_explicit():
    with pytest.raises(ValueError,match='cycles'):
        DynamicRegraspEnv(cycles=4)
