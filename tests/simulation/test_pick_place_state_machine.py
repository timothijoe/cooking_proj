import numpy as np
from dataclasses import replace

from twin_sim.grasp import GraspObservation
from twin_sim.tasks.pick_place import (
    PickPlacePhase,
    PickPlaceResult,
    PickPlaceSample,
    _completion_failure_reason,
)


def test_phase_order_is_explicit_and_complete():
    assert [phase.value for phase in PickPlacePhase] == [
        "initialize",
        "open_hand",
        "pregrasp",
        "approach",
        "close_hand",
        "stabilize",
        "lift",
        "transfer",
        "lower",
        "release",
        "retreat",
        "complete",
        "aborted",
    ]


def test_abort_result_preserves_phase_and_reason():
    result = PickPlaceResult(
        success=False,
        final_phase=PickPlacePhase.ABORTED,
        abort_phase=PickPlacePhase.STABILIZE,
        reason="grasp timeout",
        samples=(),
        placed_in_target=False,
        used_hidden_attachment=False,
    )
    assert not result.success
    assert result.abort_phase is PickPlacePhase.STABILIZE
    assert not result.used_hidden_attachment


def test_samples_copy_mutable_arrays():
    source = np.zeros(3)
    sample = PickPlaceSample.minimal(
        time_s=0.0,
        phase=PickPlacePhase.INITIALIZE,
        palm_position=source,
        cube_position=source,
    )
    source[:] = 1.0
    np.testing.assert_array_equal(sample.palm_actual_position, np.zeros(3))
    np.testing.assert_array_equal(sample.cube_position, np.zeros(3))


def test_completion_gate_rejects_cube_still_touching_hand():
    samples = [
        PickPlaceSample.minimal(
            time_s=0.0,
            phase=PickPlacePhase.INITIALIZE,
            palm_position=np.zeros(3),
            cube_position=np.array((0.0, 0.0, 0.30)),
        ),
        PickPlaceSample.minimal(
            time_s=1.0,
            phase=PickPlacePhase.LIFT,
            palm_position=np.zeros(3),
            cube_position=np.array((0.0, 0.0, 0.39)),
        ),
        PickPlaceSample.minimal(
            time_s=2.0,
            phase=PickPlacePhase.TRANSFER,
            palm_position=np.zeros(3),
            cube_position=np.array((0.20, 0.0, 0.39)),
        ),
    ]
    samples.extend(
        PickPlaceSample.minimal(
            time_s=3.0 + index * 0.01,
            phase=PickPlacePhase.COMPLETE,
            palm_position=np.zeros(3),
            cube_position=np.array((0.20, 0.0, 0.30)),
        )
        for index in range(30)
    )
    assert (
        _completion_failure_reason(
            samples,
            target_position=np.array((0.20, 0.0, 0.30)),
            target_radius_m=0.045,
        )
        is None
    )

    samples[-1] = replace(
        samples[-1],
        grasp=GraspObservation(1, False, 0.0, 0.0, 0.0),
    )
    assert "independently" in _completion_failure_reason(
        samples,
        target_position=np.array((0.20, 0.0, 0.30)),
        target_radius_m=0.045,
    )

    samples[-10] = replace(
        samples[-10],
        grasp=GraspObservation(0, False, 0.0, 0.0, 0.0),
    )
    samples[1] = replace(samples[1], support_penetration_m=0.0021)
    assert "penetrated" in _completion_failure_reason(
        samples,
        target_position=np.array((0.20, 0.0, 0.30)),
        target_radius_m=0.045,
    )

    samples[-1] = replace(
        samples[-1],
        grasp=GraspObservation(0, False, 0.0, 0.0, 0.0),
    )
    samples[1] = replace(samples[1], support_penetration_m=0.0)
    samples[-10] = replace(
        samples[-10],
        grasp=GraspObservation(1, False, 0.0, 0.0, 0.0),
    )
    assert "independently" in _completion_failure_reason(
        samples,
        target_position=np.array((0.20, 0.0, 0.30)),
        target_radius_m=0.045,
    )
