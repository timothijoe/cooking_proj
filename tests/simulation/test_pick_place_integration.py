import numpy as np
import mujoco

from twin_sim.robot import RightArmRobot
from twin_sim.tasks.pick_place import PickPlacePhase, PickPlaceTask


def test_real_contact_pick_place_meets_acceptance_contract():
    class CaptureTrace:
        def __init__(self):
            self.records = []

        def append(self, **record):
            self.records.append(record)

    trace = CaptureTrace()
    robot = RightArmRobot(viewer=False)
    right_target_before = robot._right_target.copy()
    result = PickPlaceTask(robot, trace=trace).run()
    assert result.success, (
        f"{result.abort_phase}: {result.reason}; "
        f"last={result.samples[-1].phase if result.samples else 'none'}"
    )
    cube = np.asarray([sample.cube_position for sample in result.samples])
    assert cube[:, 2].max() - cube[0, 2] >= 0.08
    assert np.linalg.norm(cube[-1, :2] - cube[0, :2]) >= 0.15
    assert result.placed_in_target
    assert not result.used_hidden_attachment
    assert max(sample.support_penetration_m for sample in result.samples) <= 0.002
    close_records = [
        record for record in trace.records if record["phase"] == "close_hand"
    ]
    ready_indices = [
        index
        for index, record in enumerate(close_records)
        if record["grasp_ready"]
    ]
    assert ready_indices == [len(close_records) - 1]
    assert not trace.records[-1]["grasp_ready"]
    assert np.linalg.norm(result.samples[-1].cube_linear_velocity) < 0.02
    assert (
        result.samples[-1].time_s - result.samples[-26].time_s
    ) >= 0.25 - 1e-9
    np.testing.assert_array_equal(robot._right_target, right_target_before)
    np.testing.assert_allclose(
        robot.joint_positions,
        right_target_before,
        atol=0.005,
        rtol=0.0,
    )
    robot.close()


def test_cube_drop_after_lift_aborts_before_transfer():
    class DropAfterLiftTask(PickPlaceTask):
        def _execute_cartesian(self, phase, *args, **kwargs):
            super()._execute_cartesian(phase, *args, **kwargs)
            if phase is PickPlacePhase.LIFT:
                joint = mujoco.mj_name2id(
                    self.robot.sim.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    "pick_cube_free",
                )
                qpos_adr = self.robot.sim.model.jnt_qposadr[joint]
                self.robot.sim.data.qpos[qpos_adr : qpos_adr + 3] = (
                    self._cube_initial
                )
                mujoco.mj_forward(
                    self.robot.sim.model,
                    self.robot.sim.data,
                )

    robot = RightArmRobot(viewer=False)
    result = DropAfterLiftTask(robot).run()

    assert not result.success
    assert result.abort_phase is PickPlacePhase.LIFT
    assert "dropped" in result.reason
    robot.close()
