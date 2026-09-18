#!/usr/bin/env python3
"""Interactive visual review of recorded chopping and the saved potato rollout.

Reference mode is a kinematic pose preview, NOT a physical cutting experiment.
Policy mode displays saved dynamic states, with the knife parked during training.
"""

import argparse
from pathlib import Path
from queue import Empty, SimpleQueue
import time

import mujoco
from mujoco import viewer
import numpy as np

from robot_core.paths import local_root
from twin_sim.potato_env import PotatoRegraspEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, default=local_root()/"outputs/potato/smoke_evaluation/evaluate-100.npz")
    parser.add_argument("--seed", type=int, default=100, help="Must match the saved rollout's seed")
    parser.add_argument("--snapshot", type=Path, help="Render a reference preview image and exit")
    args = parser.parse_args()
    with np.load(args.rollout, allow_pickle=False) as archive:
        states = archive["qpos"].copy()
        metrics = archive["metrics"].copy()
    with PotatoRegraspEnv(disturbance=False) as env:
        env.reset(seed=args.seed)
        if states.ndim != 2 or states.shape[1] != env.model.nq or len(states) != len(metrics):
            raise ValueError("Saved rollout does not match the current scene")
        initial = env.data.qpos.copy()
        state = {"mode": 0, "playing": True, "clock": 0.0, "speed": .5, "camera": 1}
        keys = SimpleQueue()

        def pose():
            clock = state["clock"]
            if state["mode"] == 0:
                env.data.qpos[:] = initial
                for ids, values in ((env.left.qpos_ids, env.left_ref),
                                    (env.right.qpos_ids, env.right_ref),
                                    (env.hand.qpos_ids, env.hand_ref)):
                    env.data.qpos[ids] = [np.interp(clock, env.times, values[:, j]) for j in range(len(ids))]
                label = "RECORDED CHOP + RETREAT | POSE PREVIEW"
                detail = "Fixed potato reference; no cutting/contact policy claims"
            else:
                index = int(np.clip(np.searchsorted(metrics[:, 0], clock), 0, len(states)-1))
                env.data.qpos[:] = states[index]
                label = "POTATO POLICY | SAVED PHYSICS | KNIFE PARKED"
                detail = f"Potato drift {metrics[index, 1]*1000:.1f} mm | hand force {metrics[index, 2]:.2f} N"
            env.data.qvel[:] = 0
            env.data.time = clock
            mujoco.mj_forward(env.model, env.data)
            return label, detail

        def camera(cam):
            cam.lookat[:] = [.435, -.025, .30]
            cam.distance = .62
            cam.azimuth, cam.elevation = {1: (165, -25), 2: (90, -80), 3: (0, -15)}[state["camera"]]

        if args.snapshot:
            state["clock"] = 1.2
            pose()
            cam = mujoco.MjvCamera()
            camera(cam)
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v3 as iio
            with mujoco.Renderer(env.model, height=480, width=640) as renderer:
                renderer.update_scene(env.data, camera=cam)
                iio.imwrite(args.snapshot, renderer.render())
            print(args.snapshot, flush=True)
            return

        with viewer.launch_passive(env.model, env.data, key_callback=keys.put,
                                   show_left_ui=False, show_right_ui=False) as window:
            camera(window.cam)
            window.opt.sitegroup[:] = 0
            print("VIEWER_READY: SPACE pause; TAB reference/policy; R restart; 1/2/3 cameras; arrows scrub/speed", flush=True)
            last = time.monotonic()
            while window.is_running():
                now = time.monotonic()
                elapsed, last = min(now-last, .1), now
                try:
                    while True:
                        key = keys.get_nowait()
                        if key == 32:
                            state["playing"] = not state["playing"]
                        elif key == 258:  # Tab
                            state["mode"] = 1-state["mode"]
                            state["clock"] = 0
                        elif key in (82, 114):
                            state["clock"] = 0
                        elif key in (49, 50, 51):
                            state["camera"] = key-48
                            with window.lock():
                                camera(window.cam)
                        elif key in (262, 263):  # Right/left
                            state["playing"] = False
                            state["clock"] = max(0, state["clock"] + (.05 if key == 262 else -.05))
                        elif key in (264, 265):
                            state["speed"] = float(np.clip(state["speed"]*(2 if key == 265 else .5), .125, 2))
                except Empty:
                    pass
                duration = float(env.times[-1] if state["mode"] == 0 else metrics[-1, 0])
                if state["playing"]:
                    state["clock"] += elapsed*state["speed"]
                    if state["clock"] > duration+.8:
                        state["clock"] = 0
                else:
                    state["clock"] = min(state["clock"], duration)
                with window.lock():
                    label, detail = pose()
                window.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    label+"\n"+detail,
                    f"{min(state['clock'], duration):.2f}/{duration:.2f}s | {state['speed']:g}x | {'PLAYING' if state['playing'] else 'PAUSED'}"),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "SPACE play/pause | TAB reference/policy | R restart\n1 side | 2 top | 3 opposite | arrows scrub/speed", "")])
                window.sync()
                time.sleep(1/60)


if __name__ == "__main__":
    main()
