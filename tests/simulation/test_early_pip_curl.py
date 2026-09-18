"""Earlier clearance should strengthen PIP flexion without changing landing."""
import numpy as np
from twin_sim.dynamic_regrasp import DynamicRegraspEnv


def test_earlier_pip_flexion_keeps_landing_geometry():
    options=dict(cycles=1,wrist_retreat_m=.008,smooth_regrasp=True,angle_threshold_deg=80,curl_release=True,landing_wrist_retreat_m=.004)
    old=DynamicRegraspEnv(**options)
    new=DynamicRegraspEnv(**options,early_pip_curl=True)
    phases=np.array(new.plan.phases)
    lift=np.where(phases=='TRIO_LIFT')[0]
    place=np.where(phases=='TRIO_PLACE')[0]
    sample=lift[int(len(lift)*.75)]
    for f in (2,3,4):
        j=new.model.jnt_qposadr[new.model.joint(f'left_finger{f}_joint3').id]
        assert new.plan.qpos[sample,j]-old.plan.qpos[sample,j]>np.radians(2)
    np.testing.assert_allclose(new.plan.tips_m[place],old.plan.tips_m[place],atol=.00005)
    arm=new.plan.left_qpos_ids
    np.testing.assert_allclose(new.plan.qpos[:,arm],old.plan.qpos[:,arm],atol=1e-12)
    move=np.where(np.isin(phases,['TRIO_LIFT','TRIO_RETREAT','TRIO_PLACE']))[0]
    assert np.max(new.plan.tips_m[move,:,2]-new.plan.tips_m[lift[0]-1,:,2])<.0125
