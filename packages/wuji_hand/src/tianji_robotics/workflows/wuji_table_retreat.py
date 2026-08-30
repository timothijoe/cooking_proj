"""Deterministic tabletop placement and retreat correction for Wuji trajectories."""

from dataclasses import dataclass

import mujoco
import numpy as np

from tianji_robotics.workflows.recorded_hand_motion import (
    detect_motion_interval,
    finger_displacement_correlations,
    smooth_joint_step_outliers,
)
from tianji_robotics.wuji_hand.models import HandTrajectory


@dataclass(frozen=True)
class TableRetreatConfig:
    retreat_distance_m: float = 0.03
    place_duration_s: float = 1.0
    retreat_duration_s: float = 2.0
    hold_duration_s: float = 2.0
    fingertip_height_m: float = 0.0015
    thumb_clearance_m: float = 0.010
    candidate_stride: int = 10
    source_frame: int | None = None


@dataclass(frozen=True)
class CorrectedHandTrajectory:
    timestamps_ns: np.ndarray
    positions_rad: np.ndarray
    palm_positions_m: np.ndarray
    palm_quaternions_wxyz: np.ndarray
    phases: tuple[str, ...]
    source_frame: int


@dataclass(frozen=True)
class LoopedTableRetreat:
    timestamps_ns: np.ndarray
    positions_rad: np.ndarray
    palm_positions_m: np.ndarray
    palm_quaternions_wxyz: np.ndarray
    phases: tuple[str, ...]
    loop_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReplayTableRetreatSummary:
    loop_count: int
    frame_count: int
    scheduled_duration_s: float
    stopped_early: bool


@dataclass(frozen=True)
class CorrectionReport:
    source_frame: int
    source_timestamp_ns: int
    candidate_score: float
    retreat_vector_m: tuple[float, float, float]
    actual_retreat_m: float
    minimum_thumb_clearance_m: float
    maximum_fingertip_height_error_m: float
    maximum_contact_slip_m: float
    maximum_hand_penetration_m: float
    maximum_retreat_fingertip_lift_m: float
    motion_start_frame: int = 0
    motion_end_frame: int = 0
    source_kind: str = "right_glove_skeleton"
    palm_down_verified: bool = False
    maximum_joint_correction_rad: float = 0.0
    rms_joint_correction_rad: float = 0.0
    per_finger_maximum_correction_rad: tuple[float, ...] = ()
    source_middle_ring_correlation: float = 0.0
    source_ring_little_correlation: float = 0.0
    corrected_middle_ring_correlation: float = 0.0
    corrected_ring_little_correlation: float = 0.0
    requested_loop_count: int = 1
    executed_loop_count: int = 0
    scheduled_playback_duration_s: float = 0.0
    stopped_early: bool = False


def build_recorded_table_retreat(
    trajectory: HandTrajectory,
    backend,
    config: TableRetreatConfig = TableRetreatConfig(),
    *,
    source_kind: str = "joint_states",
):
    """Place and retreat the palm while retaining recorded finger samples."""
    _validate_config(config)
    source = np.asarray(trajectory.positions_rad, dtype=float)
    ranges = np.asarray(list(backend.joint_ranges_rad.values()))
    corrected_joints = np.clip(source, ranges[:, 0], ranges[:, 1])
    corrected_joints = smooth_joint_step_outliers(corrected_joints, limit_rad=0.12)
    correction = corrected_joints - source
    interval = detect_motion_interval(trajectory)
    quaternion = _calibrated_palm_down_quaternion(backend, corrected_joints[0])

    backend.set_kinematic_pose(corrected_joints[0], np.zeros(3), quaternion)
    tips = backend.fingertip_positions_m()
    base_palm = np.array(
        [0.0, 0.0, backend.table_height_m + 0.003 - float(tips[:, 2].min())]
    )
    base_palm = _project_above_table(
        backend, corrected_joints[0], base_palm, quaternion
    )
    backend.set_kinematic_pose(corrected_joints[0], base_palm, quaternion)
    thumb_height = backend.thumb_position_m()[2] - backend.table_height_m
    if thumb_height < config.thumb_clearance_m:
        base_palm[2] += config.thumb_clearance_m - thumb_height

    palms = []
    phases = []
    max_penetration = 0.0
    min_thumb = float("inf")
    for index, joints in enumerate(corrected_joints):
        if index < interval.start_frame:
            fraction = 0.0
            phase = "PREPARE"
        elif index >= interval.end_frame:
            fraction = 1.0
            phase = "HOLD"
        else:
            raw = (index - interval.start_frame) / max(
                1, interval.end_frame - interval.start_frame
            )
            fraction = raw * raw * (3.0 - 2.0 * raw)
            phase = "RETREAT"
        palm = base_palm + np.array([config.retreat_distance_m * fraction, 0.0, 0.0])
        if phase == "PREPARE":
            palm = _settle_to_table_contact(backend, joints, palm, quaternion)
        else:
            palm = _project_above_table(backend, joints, palm, quaternion)
        backend.set_kinematic_pose(joints, palm, quaternion)
        thumb = backend.thumb_position_m()[2] - backend.table_height_m
        if thumb < config.thumb_clearance_m:
            palm[2] += config.thumb_clearance_m - thumb
            backend.set_kinematic_pose(joints, palm, quaternion)
        max_penetration = max(max_penetration, backend.maximum_table_penetration_m())
        min_thumb = min(
            min_thumb,
            float(backend.thumb_position_m()[2] - backend.table_height_m),
        )
        palms.append(palm)
        phases.append(phase)

    palms_array = np.asarray(palms)
    quaternions = np.tile(quaternion, (len(corrected_joints), 1))
    source_correlations = finger_displacement_correlations(source)
    corrected_correlations = finger_displacement_correlations(corrected_joints)
    palmar_below = bool(
        backend.palmar_reference_position_m()[2]
        < backend.dorsal_reference_position_m()[2]
    )
    per_finger = tuple(
        float(np.max(np.abs(correction[:, base : base + 4])))
        for base in range(0, 20, 4)
    )
    timestamps = trajectory.timestamps_ns - trajectory.timestamps_ns[0]
    corrected = CorrectedHandTrajectory(
        timestamps,
        corrected_joints,
        palms_array,
        quaternions,
        tuple(phases),
        interval.start_frame,
    )
    report = CorrectionReport(
        source_frame=interval.start_frame,
        source_timestamp_ns=int(trajectory.timestamps_ns[interval.start_frame]),
        candidate_score=0.0,
        retreat_vector_m=(config.retreat_distance_m, 0.0, 0.0),
        actual_retreat_m=float(
            np.linalg.norm(
                palms_array[interval.end_frame, :2]
                - palms_array[interval.start_frame, :2]
            )
        ),
        minimum_thumb_clearance_m=min_thumb,
        maximum_fingertip_height_error_m=0.0,
        maximum_contact_slip_m=0.0,
        maximum_hand_penetration_m=max_penetration,
        maximum_retreat_fingertip_lift_m=0.0,
        motion_start_frame=interval.start_frame,
        motion_end_frame=interval.end_frame,
        source_kind=source_kind,
        palm_down_verified=palmar_below,
        maximum_joint_correction_rad=float(np.max(np.abs(correction))),
        rms_joint_correction_rad=float(np.sqrt(np.mean(correction * correction))),
        per_finger_maximum_correction_rad=per_finger,
        source_middle_ring_correlation=source_correlations["middle_ring"],
        source_ring_little_correlation=source_correlations["ring_little"],
        corrected_middle_ring_correlation=corrected_correlations["middle_ring"],
        corrected_ring_little_correlation=corrected_correlations["ring_little"],
    )
    if not palmar_below or min_thumb < config.thumb_clearance_m or max_penetration > 0.0005:
        raise ValueError("recorded table retreat failed palm or collision preflight")
    return corrected, report


def build_looped_table_retreat(
    corrected: CorrectedHandTrajectory,
    backend,
    loops: int = 3,
) -> LoopedTableRetreat:
    """Build repeated forward gestures with preflighted reset transitions."""
    if isinstance(loops, bool) or not isinstance(loops, int) or loops < 1:
        raise ValueError("loops must be a positive integer")

    relative_ns = corrected.timestamps_ns - corrected.timestamps_ns[0]
    timestep_ns = int(round(backend.timestep_s * 1e9))
    timestamps: list[int] = []
    joints_frames: list[np.ndarray] = []
    palm_frames: list[np.ndarray] = []
    quaternion_frames: list[np.ndarray] = []
    phases: list[str] = []
    loop_indices: list[int] = []

    def append_forward(loop_index: int) -> None:
        start_ns = 0 if not timestamps else timestamps[-1] + timestep_ns
        timestamps.extend((start_ns + relative_ns).astype(np.int64).tolist())
        joints_frames.extend(np.asarray(corrected.positions_rad))
        palm_frames.extend(np.asarray(corrected.palm_positions_m))
        quaternion_frames.extend(np.asarray(corrected.palm_quaternions_wxyz))
        phases.extend(corrected.phases)
        loop_indices.extend([loop_index] * len(corrected.timestamps_ns))

    append_forward(0)
    for loop_index in range(1, loops):
        final_joints = np.asarray(joints_frames[-1], dtype=float)
        final_palm = np.asarray(palm_frames[-1], dtype=float)
        final_quaternion = np.asarray(quaternion_frames[-1], dtype=float)
        initial_joints = np.asarray(corrected.positions_rad[0], dtype=float)
        initial_palm = np.asarray(corrected.palm_positions_m[0], dtype=float)
        initial_quaternion = np.asarray(corrected.palm_quaternions_wxyz[0], dtype=float)
        reset_steps = max(
            1,
            int(np.ceil(0.5 / backend.timestep_s)),
            int(np.ceil(np.max(np.abs(initial_joints - final_joints)) / 0.12)),
            int(np.ceil(np.linalg.norm(initial_palm - final_palm) / 0.001)),
        )
        for fraction in np.linspace(0.0, 1.0, reset_steps + 1)[1:]:
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            joints = final_joints + smooth * (initial_joints - final_joints)
            palm = final_palm + smooth * (initial_palm - final_palm)
            quaternion = final_quaternion + smooth * (
                initial_quaternion - final_quaternion
            )
            quaternion /= np.linalg.norm(quaternion)
            palm = _project_above_table(backend, joints, palm, quaternion)
            backend.set_kinematic_pose(joints, palm, quaternion)
            thumb = backend.thumb_position_m()[2] - backend.table_height_m
            if thumb < 0.010:
                palm[2] += 0.010 - thumb
                backend.set_kinematic_pose(joints, palm, quaternion)
            if backend.maximum_table_penetration_m() > 0.0005:
                raise ValueError("reset trajectory crosses the table")
            timestamps.append(timestamps[-1] + timestep_ns)
            joints_frames.append(joints)
            palm_frames.append(palm)
            quaternion_frames.append(quaternion)
            phases.append("RESET")
            loop_indices.append(loop_index - 1)
        append_forward(loop_index)

    return LoopedTableRetreat(
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        positions_rad=np.asarray(joints_frames),
        palm_positions_m=np.asarray(palm_frames),
        palm_quaternions_wxyz=np.asarray(quaternion_frames),
        phases=tuple(phases),
        loop_indices=tuple(loop_indices),
    )


def build_table_retreat(trajectory: HandTrajectory, backend, config: TableRetreatConfig = TableRetreatConfig()):
    _validate_config(config)
    source_frame, score = _select_frame(trajectory, backend, config)
    selected = trajectory.positions_rad[source_frame].copy()
    ranges = np.asarray(list(backend.joint_ranges_rad.values()))
    place_seed = selected.copy()
    place_seed[4:] = np.clip(0.0, ranges[4:, 0], ranges[4:, 1])
    quaternion = _fit_palm_down_quaternion(backend, place_seed)
    place_joints, place_palm = _solve_place(
        backend, place_seed, quaternion, config
    )
    backend.set_kinematic_pose(place_joints, place_palm, quaternion)
    contact_targets = backend.fingertip_positions_m().copy()
    if backend.thumb_position_m()[2] - backend.table_height_m < config.thumb_clearance_m:
        raise ValueError("selected pose cannot keep the thumb clear of the table")

    dt = backend.timestep_s
    place_count = _steps(config.place_duration_s, dt)
    retreat_count = _steps(config.retreat_duration_s, dt)
    hold_count = _steps(config.hold_duration_s, dt)
    joints_frames=[]; palm_frames=[]; phases=[]
    hover_palm = place_palm + np.array([0.0, 0.0, 0.05])
    for fraction in np.linspace(0.0, 1.0, place_count + 1):
        joints_frames.append(place_seed + fraction * (place_joints - place_seed))
        palm_frames.append(hover_palm + fraction * (place_palm - hover_palm))
        phases.append("PLACE")
    retreat_vector = np.array([config.retreat_distance_m, 0.0, 0.0])
    curl_target = place_joints.copy()
    ranges = np.asarray(list(backend.joint_ranges_rad.values()))
    for base in (4, 8, 12, 16):
        curl_target[base] += 0.35
        curl_target[base + 2] += 0.35
        curl_target[base + 3] += 0.25
    curl_target = np.clip(curl_target, ranges[:, 0], ranges[:, 1])
    previous = place_joints
    for fraction in np.linspace(0.0, 1.0, retreat_count + 1)[1:]:
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        solved = place_joints + smooth * (curl_target - place_joints)
        previous = previous + np.clip(solved - previous, -0.1, 0.1)
        palm = place_palm + smooth * retreat_vector
        palm = _project_above_table(backend, previous, palm, quaternion)
        joints_frames.append(previous.copy()); palm_frames.append(palm); phases.append("RETREAT")
    final_palm = palm_frames[-1]
    for _ in range(hold_count):
        joints_frames.append(previous.copy()); palm_frames.append(final_palm.copy()); phases.append("HOLD")
    joints = np.asarray(joints_frames); palms=np.asarray(palm_frames)
    quaternions=np.tile(quaternion,(len(joints),1))
    diagnostics = _preflight(backend, joints, palms, quaternions, phases, contact_targets, config)
    timestamps = np.arange(len(joints), dtype=np.int64) * int(round(dt * 1e9))
    corrected=CorrectedHandTrajectory(timestamps,joints,palms,quaternions,tuple(phases),source_frame)
    report=CorrectionReport(source_frame,int(trajectory.timestamps_ns[source_frame]),score,tuple(retreat_vector),float(np.linalg.norm(retreat_vector)),*diagnostics)
    return corrected, report


def replay_table_retreat(
    corrected: LoopedTableRetreat,
    backend,
) -> ReplayTableRetreatSummary:
    timestamps = np.asarray(corrected.timestamps_ns, dtype=np.int64)
    first_timestamp_ns = int(timestamps[0])
    executed_steps = 0
    executed_frames = 0
    has_viewer = bool(getattr(backend, "has_viewer", False))

    def viewer_closed() -> bool:
        return has_viewer and not backend.viewer_is_running()

    def completed_loops(frame_count: int, stopped: bool) -> int:
        prefix = corrected.phases[:frame_count]
        reset_starts = sum(
            phase == "RESET" and (index == 0 or prefix[index - 1] != "RESET")
            for index, phase in enumerate(prefix)
        )
        return reset_starts + (0 if stopped or frame_count == 0 else 1)

    for frame_index, (joints, palm, quaternion) in enumerate(
        zip(
            corrected.positions_rad,
            corrected.palm_positions_m,
            corrected.palm_quaternions_wxyz,
            strict=True,
        )
    ):
        if viewer_closed():
            break
        elapsed_s = float(int(timestamps[frame_index]) - first_timestamp_ns) / 1e9
        target_steps = round(elapsed_s / backend.timestep_s)
        for _ in range(max(0, target_steps - executed_steps)):
            if viewer_closed():
                return ReplayTableRetreatSummary(
                    loop_count=completed_loops(executed_frames, True),
                    frame_count=executed_frames,
                    scheduled_duration_s=executed_steps * backend.timestep_s,
                    stopped_early=True,
                )
            backend.step(backend.timestep_s)
            executed_steps += 1
        if viewer_closed():
            break
        backend.command_pose(joints, palm, quaternion)
        executed_frames += 1

    stopped_early = executed_frames < len(timestamps)
    duration_s = (
        executed_steps * backend.timestep_s
        if stopped_early
        else float(timestamps[-1] - timestamps[0]) / 1e9
    )
    return ReplayTableRetreatSummary(
        loop_count=completed_loops(executed_frames, stopped_early),
        frame_count=executed_frames,
        scheduled_duration_s=duration_s,
        stopped_early=stopped_early,
    )


def _select_frame(trajectory, backend, config):
    candidates = [config.source_frame] if config.source_frame is not None else list(range(0, len(trajectory.positions_rad), config.candidate_stride))
    if candidates[-1] != len(trajectory.positions_rad)-1 and config.source_frame is None: candidates.append(len(trajectory.positions_rad)-1)
    best=None
    for index in candidates:
        if index is None or not 0 <= index < len(trajectory.positions_rad): raise ValueError("source frame is outside trajectory")
        backend.set_kinematic_pose(trajectory.positions_rad[index], np.zeros(3), [1,0,0,0])
        tips=backend.fingertip_positions_m(); thumb=backend.thumb_position_m()
        relative_thumb=float(thumb[2]-tips[:,2].mean()+config.fingertip_height_m)
        if relative_thumb < config.thumb_clearance_m: continue
        score=float(np.ptp(tips[:,2]) + 0.1*np.std(tips[:,:2]))
        candidate=(score,index)
        if best is None or candidate < best: best=candidate
    if best is None: raise ValueError("no feasible source frame keeps the thumb clear of the table")
    return best[1],best[0]


def _solve_contacts(backend, initial, palm, quaternion, targets, *, vertical_only):
    q=np.asarray(initial,dtype=float).copy(); active=np.arange(4,20)
    ranges=np.asarray(list(backend.joint_ranges_rad.values()))
    for _ in range(40):
        backend.set_kinematic_pose(q,palm,quaternion)
        current=backend.fingertip_positions_m(); error=(targets-current)
        if vertical_only:
            residual=error[:,2]; jacobian=backend.fingertip_position_jacobian()[2::3,active]
        else:
            residual=error.reshape(-1); jacobian=backend.fingertip_position_jacobian()[:,active]
        if np.max(np.abs(residual)) < 0.001: break
        lhs=jacobian.T@jacobian + 1e-4*np.eye(len(active))
        step=np.linalg.solve(lhs,jacobian.T@residual)
        q[active]+=np.clip(step,-0.04,0.04)
        q=np.clip(q,ranges[:,0],ranges[:,1])
    return q


def _fit_palm_down_quaternion(backend, joints) -> np.ndarray:
    backend.set_kinematic_pose(joints, np.array([0.0, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    tips = backend.fingertip_positions_m()
    roots = backend.long_finger_root_positions_m()
    forward = tips.mean(axis=0) - roots.mean(axis=0)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-8:
        raise ValueError("cannot fit palm frame from coincident finger landmarks")
    forward /= forward_norm
    lateral = roots[-1] - roots[0]
    lateral -= forward * float(np.dot(lateral, forward))
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm <= 1e-8:
        raise ValueError("cannot fit palm frame from collinear finger landmarks")
    lateral /= lateral_norm
    normal = np.cross(forward, lateral)
    local_basis = np.column_stack((forward, lateral, normal))
    # Fingers extend toward world -X, span toward +Y, and the palm-facing
    # normal points toward -Z so the knuckles remain above the table.
    world_basis = np.column_stack(([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]))
    rotation = world_basis @ local_basis.T
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
    return quaternion


def _calibrated_palm_down_quaternion(backend, joints) -> np.ndarray:
    origin = np.array([0.0, 0.0, 0.2])
    backend.set_kinematic_pose(joints, origin, np.array([1.0, 0.0, 0.0, 0.0]))
    roots = backend.long_finger_root_positions_m()
    # Palm forward is the fixed wrist-to-finger-root direction. The previous
    # implementation used root-to-tip, which tilts with recorded finger curl.
    forward = roots.mean(axis=0) - origin
    forward /= np.linalg.norm(forward)
    lateral = roots[-1] - roots[0]
    lateral -= forward * float(np.dot(lateral, forward))
    lateral /= np.linalg.norm(lateral)
    normal = np.cross(forward, lateral)
    local_basis = np.column_stack((forward, lateral, normal))
    candidates = (
        np.column_stack(([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0])),
        np.column_stack(([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0])),
    )
    for world_basis in candidates:
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, (world_basis @ local_basis.T).reshape(-1))
        backend.set_kinematic_pose(joints, origin, quaternion)
        if backend.palmar_reference_position_m()[2] < backend.dorsal_reference_position_m()[2]:
            return quaternion
    raise ValueError("official Wuji palm-side calibration is inverted")


def _solve_place(backend, initial, quaternion, config):
    q = np.asarray(initial, dtype=float).copy()
    backend.set_kinematic_pose(q, np.zeros(3), quaternion)
    tips = backend.fingertip_positions_m()
    provisional_height = 0.003
    palm = np.array([0.0, 0.0, provisional_height - float(tips[:, 2].mean())])
    provisional_targets = tips.copy()
    provisional_targets[:, 2] = backend.table_height_m + provisional_height
    q = _solve_contacts(backend, q, palm, quaternion, provisional_targets, vertical_only=True)
    backend.set_kinematic_pose(q, palm, quaternion)
    distances = backend.long_fingertip_table_distances_m()
    if not np.isfinite(distances).all():
        raise ValueError("selected pose cannot calibrate all four fingertip surfaces")
    surface_offsets = backend.fingertip_positions_m()[:, 2] - distances
    targets = backend.fingertip_positions_m().copy()
    targets[:, 2] = surface_offsets + 0.0002
    for _ in range(30):
        q = _solve_contacts(backend, q, palm, quaternion, targets, vertical_only=True)
        backend.set_kinematic_pose(q, palm, quaternion)
        penetration = backend.maximum_table_penetration_m()
        if penetration <= 0.0005:
            if backend.thumb_position_m()[2] - backend.table_height_m < config.thumb_clearance_m:
                raise ValueError("selected pose cannot keep the thumb clear of the table")
            return q, palm
        palm[2] += penetration + 0.0001
    raise ValueError("selected pose cannot place the whole hand above the table")


def _project_above_table(backend, joints, palm, quaternion):
    corrected = np.asarray(palm, dtype=float).copy()
    for _ in range(4):
        backend.set_kinematic_pose(joints, corrected, quaternion)
        penetration = backend.maximum_table_penetration_m()
        if penetration <= 0.0005:
            return corrected
        corrected[2] += penetration - 0.0005 + 0.0001
    raise ValueError("retreat frame cannot be projected above the table")


def _settle_to_table_contact(backend, joints, palm, quaternion):
    """Lower a prepare pose to first contact without exceeding tolerance."""
    corrected = np.asarray(palm, dtype=float).copy()
    for _ in range(1000):
        backend.set_kinematic_pose(joints, corrected, quaternion)
        clearance = backend.minimum_hand_table_clearance_m()
        if np.isfinite(clearance):
            if clearance < -0.0005:
                corrected[2] += -0.0004 - clearance
                backend.set_kinematic_pose(joints, corrected, quaternion)
            if backend.thumb_position_m()[2] - backend.table_height_m < 0.010:
                raise ValueError("prepare contact would place the thumb on the table")
            return corrected
        corrected[2] -= 0.00025
    raise ValueError("prepare pose cannot reach the table")


def _preflight(backend,joints,palms,quaternions,phases,targets,config):
    if not np.isfinite(joints).all() or np.max(np.abs(np.diff(joints,axis=0))) > 0.12: raise ValueError("corrected trajectory exceeds joint-step limit")
    min_thumb=float("inf"); max_height=0.0; max_slip=0.0; max_penetration=0.0; max_lift=0.0
    place_end=phases.index("RETREAT") if "RETREAT" in phases else len(phases)
    place_tip_heights = None
    for i,(q,palm,quat) in enumerate(zip(joints,palms,quaternions,strict=True)):
        backend.set_kinematic_pose(q,palm,quat)
        min_thumb=min(min_thumb,float(backend.thumb_position_m()[2]-backend.table_height_m))
        max_penetration=max(max_penetration,backend.maximum_table_penetration_m())
        if i == place_end - 1:
            place_tip_heights=backend.fingertip_positions_m()[:,2].copy()
        if i >= place_end:
            tips=backend.fingertip_positions_m(); max_height=max(max_height,float(np.max(np.abs(tips[:,2]-(backend.table_height_m+config.fingertip_height_m))))); max_slip=max(max_slip,float(np.max(np.linalg.norm(tips[:,:2]-targets[:,:2],axis=1))))
            if place_tip_heights is not None:
                max_lift=max(max_lift,float(np.max(tips[:,2]-place_tip_heights)))
    if min_thumb < config.thumb_clearance_m or max_penetration > 0.0005: raise ValueError(f"contact correction failed preflight: thumb={min_thumb}, penetration={max_penetration}")
    return min_thumb,max_height,max_slip,max_penetration,max_lift


def _steps(duration,dt):
    return max(1,int(round(duration/dt)))


def _validate_config(config):
    values=(config.retreat_distance_m,config.place_duration_s,config.retreat_duration_s,config.hold_duration_s,config.fingertip_height_m,config.thumb_clearance_m)
    if not np.isfinite(values).all() or any(value<=0 for value in values) or config.candidate_stride<1: raise ValueError("table-retreat configuration must be positive and finite")
