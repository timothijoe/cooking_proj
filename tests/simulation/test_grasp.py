import mujoco
import numpy as np

from twin_sim.grasp import GraspMonitor, GraspObservation
from twin_sim.model import SimulationModel


def observation(*, opposing=True, speed=0.0, force=2.0):
    return GraspObservation(
        hand_contact_count=2,
        opposing_contacts=opposing,
        cube_linear_speed_m_s=speed,
        cube_angular_speed_rad_s=0.0,
        max_hand_actuator_force=force,
    )


def test_grasp_requires_continuous_stable_dwell():
    monitor = GraspMonitor(stable_dwell_s=0.10)
    for _ in range(9):
        monitor.update(observation(), 0.01)
    assert not monitor.ready
    monitor.update(observation(), 0.01)
    assert monitor.ready


def test_unstable_sample_resets_dwell_and_force_aborts():
    monitor = GraspMonitor(stable_dwell_s=0.10, abort_force=20.0)
    monitor.update(observation(), 0.05)
    monitor.update(observation(speed=0.2), 0.01)
    assert not monitor.ready
    monitor.update(observation(force=21.0), 0.01)
    assert monitor.abort_reason == "hand force limit exceeded"


def test_contact_observation_is_finite_and_cube_specific():
    sim = SimulationModel.load()
    monitor = GraspMonitor()
    for _ in range(20):
        mujoco.mj_step(sim.model, sim.data)
    value = monitor.observe(sim)
    assert value.hand_contact_count == 0
    assert not value.opposing_contacts
    assert np.isfinite(
        (
            value.cube_linear_speed_m_s,
            value.cube_angular_speed_rad_s,
            value.max_hand_actuator_force,
        )
    ).all()


def test_opposing_contacts_must_come_from_different_hand_parts():
    normal = np.asarray((1.0, 0.0, 0.0))
    opposite = -normal

    assert not GraspMonitor._has_opposing_parts(
        ((12, normal), (12, opposite))
    )
    assert GraspMonitor._has_opposing_parts(
        ((12, normal), (18, opposite))
    )
