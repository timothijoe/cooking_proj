"""Local MuJoCo backend for interactive Wuji hand angle-bar control."""

from __future__ import annotations

import datetime
import os
import subprocess
import tempfile
import time
import tkinter as tk
import tkinter.filedialog as tkfiledialog
import tkinter.messagebox as tkmsg
from collections.abc import Callable

import mujoco
import numpy as np

from tianji_robotics.simulation.paths import official_wuji_hand_mjcf


_SIDES = frozenset({"left", "right"})
_FINGER_LABELS = ("Thumb", "Index", "Middle", "Ring", "Little")
PANEL_SAFETY_TEXT = "MuJoCo-only simulation — does not command hardware."

# Path to the wujihand venv python (for sending poses to the real hand)
_WUJIHAND_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    ".venv-wujihand", "bin", "python",
)
_SEND_POSE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "scripts", "send_pose_to_wuji_hand.py",
)
# Default save directory for recordings
_RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "recordings", "wuji",
)
OPEN_TARGET_RAD = {
    # Explicit per-side poses, kept separate so a future handed model can
    # change one pose without silently changing the other.
    "left": np.asarray((0.15, 0.0, 0.05, 0.05) + (0.05, 0.0, 0.05, 0.05) * 4, dtype=float),
    "right": np.asarray((0.15, 0.0, 0.05, 0.05) + (0.05, 0.0, 0.05, 0.05) * 4, dtype=float),
}


def _validate_side(side: str) -> str:
    if side not in _SIDES:
        raise ValueError("hand side must be 'left' or 'right'")
    return side


class LocalWujiHand:
    """Control one official 20-DOF Wuji hand entirely within MuJoCo."""

    def __init__(self, side: str, *, viewer: bool = False) -> None:
        self.side = _validate_side(side)
        self.model = mujoco.MjModel.from_xml_path(str(official_wuji_hand_mjcf(self.side)))
        self.data = mujoco.MjData(self.model)
        self._actuator_ids = np.arange(self.model.nu, dtype=np.int32)
        if self.model.nu != 20:
            raise RuntimeError("official Wuji hand must expose 20 actuators")
        self.joint_names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in self._actuator_ids
        )
        if any(name is None for name in self.joint_names):
            raise RuntimeError("official Wuji hand actuators must be named")
        self.control_ranges_rad = self.model.actuator_ctrlrange[self._actuator_ids].copy()
        self._open_target_rad = OPEN_TARGET_RAD[self.side].copy()
        if (
            np.any(self._open_target_rad < self.control_ranges_rad[:, 0])
            or np.any(self._open_target_rad > self.control_ranges_rad[:, 1])
        ):
            raise RuntimeError(f"{self.side} open target is outside model control ranges")
        self._closed = False
        self._viewer = None
        if viewer:
            from mujoco import viewer as mujoco_viewer

            self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        self.command(self._open_target_rad)

    @property
    def target_rad(self) -> np.ndarray:
        self._require_open()
        return self.data.ctrl[self._actuator_ids].copy()

    @property
    def open_target_rad(self) -> np.ndarray:
        """Return the validated, side-specific open hand target in radians."""
        self._require_open()
        return self._open_target_rad.copy()

    @property
    def viewer_is_running(self) -> bool:
        """Whether this backend owns a passive Viewer that remains open."""
        return self._viewer is not None and self._viewer.is_running()

    @property
    def timestep_s(self) -> float:
        """Return the duration advanced by one :meth:`step` call."""
        self._require_open()
        return float(self.model.opt.timestep)

    def command(self, target_rad: np.ndarray) -> None:
        self._require_open()
        target = np.asarray(target_rad, dtype=float)
        if target.shape != (20,) or not np.isfinite(target).all():
            raise ValueError("hand target must contain 20 finite radians")
        if np.any(target < self.control_ranges_rad[:, 0]) or np.any(
            target > self.control_ranges_rad[:, 1]
        ):
            raise ValueError("hand target is outside range")
        self.data.ctrl[self._actuator_ids] = target

    def reset_open(self) -> None:
        self.command(self._open_target_rad)

    def step(self) -> None:
        self._require_open()
        mujoco.mj_step(self.model, self.data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._closed:
            return
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.close()
        self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("local Wuji hand backend is closed")


class WujiAngleBarPanel:
    """Tk controls that command one locally simulated Wuji hand at a time."""

    def __init__(
        self,
        root: tk.Tk,
        backend_factory: Callable[..., LocalWujiHand] = LocalWujiHand,
        initial_side: str = "left",
    ) -> None:
        self._root = root
        self._backend_factory = backend_factory
        self._backend = backend_factory(initial_side, viewer=True)
        self._closed = False
        self._side = tk.StringVar(value=initial_side)
        self._targets = [tk.DoubleVar(value=value) for value in self._backend.target_rad]
        self._slider_groups: tk.Frame | None = None
        self._value_labels: list[tk.Label] = []
        self._recording = False
        self._recorded_frames: list[np.ndarray] = []
        self._recorded_timestamps: list[int] = []
        self._status_var = tk.StringVar(value="")
        self._build_controls()
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._root.after(0, self._tick)

    def _build_controls(self) -> None:
        self._root.title("Wuji hand local angle control")
        controls = tk.Frame(self._root, padx=10, pady=10)
        controls.pack(fill=tk.X)
        tk.Button(controls, text="Save Pose", command=self.save_pose).pack(side=tk.LEFT)
        self._record_button = tk.Button(controls, text="Record", command=self.toggle_record)
        self._record_button.pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(controls, text="Send to Hand", command=self.send_to_hand).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        tk.Label(controls, text="Hand side:").pack(side=tk.LEFT, padx=(20, 0))
        for side in ("left", "right"):
            tk.Radiobutton(
                controls,
                text=side.capitalize(),
                variable=self._side,
                value=side,
                command=lambda selected=side: self.switch_side(selected),
            ).pack(side=tk.LEFT)
        tk.Button(controls, text="Reset open hand", command=self.reset_open).pack(side=tk.RIGHT)
        tk.Label(
            controls,
            text=PANEL_SAFETY_TEXT,
            anchor=tk.W,
        ).pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))
        tk.Label(
            controls, textvariable=self._status_var, anchor=tk.W, fg="#444444"
        ).pack(fill=tk.X, side=tk.BOTTOM)
        self._slider_host = tk.Frame(self._root, padx=10, pady=10)
        self._slider_host.pack(fill=tk.BOTH, expand=True)
        self._rebuild_slider_groups()

    def _rebuild_slider_groups(self) -> None:
        if self._slider_groups is not None:
            self._slider_groups.destroy()
        self._slider_groups = tk.Frame(self._slider_host)
        self._slider_groups.pack(fill=tk.BOTH, expand=True)
        self._value_labels = []
        for finger_index, finger_name in enumerate(_FINGER_LABELS):
            group = tk.LabelFrame(self._slider_groups, text=finger_name, padx=6, pady=4)
            group.pack(fill=tk.X, pady=3)
            for joint_offset in range(4):
                index = finger_index * 4 + joint_offset
                self._build_slider_row(group, index)

    def _build_slider_row(self, parent: tk.Misc, index: int) -> None:
        lower, upper = self._backend.control_ranges_rad[index]
        tk.Label(parent, text=self._backend.joint_names[index], width=24, anchor=tk.W).grid(
            row=index % 4, column=0, sticky=tk.W
        )
        scale = tk.Scale(
            parent,
            from_=float(lower),
            to=float(upper),
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self._targets[index],
            command=lambda _value, selected=index: self._apply_slider(selected),
            length=300,
        )
        scale.grid(row=index % 4, column=1, sticky=tk.EW)
        value_label = tk.Label(parent, text=self._format_radians(self._targets[index].get()), width=10)
        value_label.grid(row=index % 4, column=2, sticky=tk.E)
        self._value_labels.append(value_label)
        parent.grid_columnconfigure(1, weight=1)

    def _apply_slider(self, index: int) -> None:
        target = self._backend.target_rad
        target[index] = self._targets[index].get()
        self._backend.command(target)
        self._value_labels[index].configure(text=self._format_radians(target[index]))

    def reset_open(self) -> None:
        self._backend.reset_open()
        for index, value in enumerate(self._backend.target_rad):
            self._targets[index].set(value)
            self._value_labels[index].configure(text=self._format_radians(value))

    def switch_side(self, side: str) -> None:
        if getattr(self, "_recording", False):
            self._stop_recording(discard=True)
        old_backend = self._backend
        self._backend = None
        try:
            if old_backend is not None:
                old_backend.close()
            self._backend = self._backend_factory(side, viewer=True)
            self._side.set(side)
            self._targets = [tk.DoubleVar(value=value) for value in self._backend.target_rad]
            self._rebuild_slider_groups()
        except Exception:
            self.close()
            raise

    def _make_timestamp(self) -> str:
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_npz(self, default_name: str, data: dict) -> str | None:
        path = tkfiledialog.asksaveasfilename(
            defaultextension=".npz",
            initialdir=_RECORDINGS_DIR,
            initialfile=default_name,
            filetypes=[("NumPy NPZ", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return None
        np.savez_compressed(path, **data)
        return path

    def save_pose(self) -> None:
        """Save the current slider pose as a single-frame NPZ file."""
        side = self._side.get()
        timestamp = self._make_timestamp()
        default_name = f"wuji_pose_{side}_{timestamp}.npz"
        positions = self._backend.target_rad.copy()  # shape (20,)
        data = {
            "joint_positions_rad": positions[np.newaxis, :],  # shape (1, 20)
            "timestamps_ns": np.array([time.time_ns()], dtype=np.int64),
            "side": side,
            "joint_names": np.array(self._backend.joint_names),
        }
        path = self._save_npz(default_name, data)
        if path:
            self._status_var.set(f"✅ Pose saved: {os.path.basename(path)}")

    def toggle_record(self) -> None:
        """Start or stop recording a trajectory."""
        if self._recording:
            self._stop_recording(discard=False)
        else:
            self._recording = True
            self._recorded_frames = [self._backend.target_rad.copy()]
            self._recorded_timestamps = [time.time_ns()]
            self._record_button.configure(text="■ Stop", fg="red")
            self._status_var.set("🔴 Recording...")

    def _stop_recording(self, *, discard: bool = False) -> None:
        if not getattr(self, "_recording", False):
            return
        self._recording = False
        record_button = getattr(self, "_record_button", None)
        if record_button is not None:
            try:
                record_button.configure(text="Record", fg="black")
            except Exception:
                pass
        if discard or not self._recorded_frames:
            self._recorded_frames = []
            self._recorded_timestamps = []
            status_var = getattr(self, "_status_var", None)
            if discard and status_var is not None:
                status_var.set("")
            return
        self._save_recording()

    def _save_recording(self) -> None:
        side = self._side.get()
        timestamp = self._make_timestamp()
        default_name = f"wuji_trajectory_{side}_{timestamp}.npz"
        positions = np.asarray(self._recorded_frames, dtype=np.float64)
        timestamps = np.asarray(self._recorded_timestamps, dtype=np.int64)
        data = {
            "joint_positions_rad": positions,
            "timestamps_ns": timestamps,
            "side": side,
            "joint_names": np.array(self._backend.joint_names),
            "frame_count": len(self._recorded_frames),
        }
        path = self._save_npz(default_name, data)
        self._recorded_frames = []
        self._recorded_timestamps = []
        if path:
            self._status_var.set(f"✅ Trajectory saved: {os.path.basename(path)} "
                                 f"({data['frame_count']} frames)")
        else:
            self._status_var.set("")

    def send_to_hand(self) -> None:
        """Send the current pose to the real Wuji hand via subprocess."""
        if not os.path.isfile(_WUJIHAND_PYTHON) or not os.path.isfile(_SEND_POSE_SCRIPT):
            self._status_var.set("❌ Send to Hand: venv or script not found")
            return

        side = self._side.get()
        if not tkmsg.askyesno(
            "Send to Real Hand",
            f"Send the current {side} hand pose to the real Wuji hand?\n\n"
            "Make sure the hand is connected and powered on.\n"
            "The hand will be enabled, moved to the target pose,\n"
            "then disabled after a few seconds.",
        ):
            return

        # Write temporary NPZ file
        timestamp = self._make_timestamp()
        tmp_path = os.path.join(tempfile.gettempdir(),
                                f"wuji_send_pose_{side}_{timestamp}.npz")
        try:
            positions = self._backend.target_rad.copy()
            data = {
                "joint_positions_rad": positions[np.newaxis, :],
                "timestamps_ns": np.array([time.time_ns()], dtype=np.int64),
                "side": side,
                "joint_names": np.array(self._backend.joint_names),
            }
            np.savez_compressed(tmp_path, **data)

            self._status_var.set(f"⏳ Sending to {side} hand...")
            self._root.update()

            result = subprocess.run(
                [_WUJIHAND_PYTHON, _SEND_POSE_SCRIPT, tmp_path],
                capture_output=True, text=True, timeout=60,
            )

            if result.returncode == 0:
                self._status_var.set("✅ Pose sent to hand successfully")
            else:
                err = result.stderr.strip() or result.stdout.strip()
                self._status_var.set(f"❌ Hand command failed: {err[:80]}")
        except subprocess.TimeoutExpired:
            self._status_var.set("❌ Hand command timed out")
        except Exception as exc:
            self._status_var.set(f"❌ Error: {exc}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _tick(self) -> None:
        try:
            if self._closed:
                return
            if not self._root.winfo_exists():
                self.close()
                return
            backend = self._backend
            if backend is None:
                self.close()
                return
            backend.step()
            if getattr(self, "_recording", False):
                self._recorded_frames.append(backend.target_rad.copy())
                self._recorded_timestamps.append(time.time_ns())
            if not backend.viewer_is_running:
                self.close()
                return
            self._root.after(self._tick_interval_ms(backend), self._tick)
        except tk.TclError:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        backend = self._backend
        self._backend = None
        try:
            if backend is not None:
                backend.close()
        finally:
            try:
                self._root.quit()
            except tk.TclError:
                pass
            try:
                self._root.destroy()
            except tk.TclError:
                pass

    @staticmethod
    def _tick_interval_ms(backend: LocalWujiHand) -> int:
        """Let Tk own real-time pacing for one-model-timestep backend steps."""
        return max(1, int(round(backend.timestep_s * 1000)))

    @staticmethod
    def _format_radians(value: float) -> str:
        return f"{value:.3f} rad"


def run_local_angle_bar(side: str) -> int:
    """Run a local MuJoCo hand angle-bar panel for the requested side."""
    _validate_side(side)
    root = tk.Tk()
    WujiAngleBarPanel(root, initial_side=side)
    root.mainloop()
    return 0
