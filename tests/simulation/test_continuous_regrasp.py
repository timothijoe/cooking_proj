"""Three sequential cutting/regrasp rounds in one physical episode."""
import numpy as np
import pytest
from twin_sim.dynamic_regrasp import DynamicRegraspEnv

@pytest.fixture(scope='module')
def rollout():
    env=DynamicRegraspEnv(cycles=3,pip_guard=True)
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
    # Geometric reference completion does not imply successful stabilization.
    assert rows[-1]['is_success']==all(c['verified'] for c in rows[-1]['cycle_checks'])
    assert len(rows[-1]['cycle_checks'])==3
    assert rows[-1]['max_displacement_m']<.04
    assert np.all(np.diff(times)>0)
    assert times[-1]>29
    # A reset at a round boundary would teleport the object. Contact impulses
    # inside a round may legitimately move it faster than this seam tolerance.
    boundaries=np.array([i for i in range(1,len(rows)) if rows[i]['cycle']!=rows[i-1]['cycle']])
    assert np.max(np.linalg.norm(positions[boundaries]-positions[boundaries-1],axis=1))<.0005
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


def test_all_three_rounds_retract_only_after_actual_angles_and_contact(rollout):
    env,rows,_,_=rollout
    events=rows[-1]['angle_gate_events']
    assert [e['cycle'] for e in events]==[1,2,3]
    assert events[0]['physical_time_s'] < env.plan.times_s[env.gate_last[0]]
    for event in events:
        first=next(i for i,row in enumerate(rows) if any(e['cycle']==event['cycle'] for e in row['angle_gate_events']))
        assert first>=2
        assert all(min(row['middle_angles_deg'])>=85 for row in rows[first-2:first+1])
    for e in events:
        assert min(e['angles_deg'])>=85
        assert min(e['main_tip_loads_n'])>.1
    for row in rows:
        if row['phase']=='KNUCKLE_BACK':
            assert any(event['cycle']==row['cycle'] for event in row['angle_gate_events'])


def test_unreached_angle_waits_then_times_out_instead_of_retracting():
    env=DynamicRegraspEnv(cycles=1)
    env.angle_threshold_deg=91  # deliberately impossible: verify the gate is real
    env.reset(seed=100)
    for _ in range(env.max_steps):
        _,_,done,truncated,info=env.step(np.zeros(4))
        assert info['phase']!='KNUCKLE_BACK'
        assert not info['angle_gate_events']
        if done or truncated:break
    assert info['termination_reason']=='angle_gate_timeout'
    assert info['completed_cycles']==0
    assert not info['is_success']


def test_reference_does_not_probe_pip_forward_before_retreat(rollout):
    env,rows,_,_=rollout
    p=env.plan;pip_y=p.tips_m[:,:,1]-p.leads_m
    for cycle in range(3):
        for phase in ('CUT_ADVANCE','ANGLE_GATE','KNUCKLE_BACK','TRIO_LIFT','TRIO_RETREAT'):
            mask=(p.cycle_ids==cycle)&(np.array(p.phases)==phase)
            values=pip_y[mask]
            if len(values):assert np.max(values[0]-values.min(axis=0))<.00015
    for cycle in (1,2,3):
        for phase in ('CUT_ADVANCE','TRIO_LIFT','TRIO_RETREAT'):
            group=[r for r in rows if r['cycle']==cycle and r['phase']==phase]
            if group:
                values=np.array([r['pip_positions_m'] for r in group])[:,:,1]
                assert np.max(values[0]-values.min(axis=0))<.0006
