#!/usr/bin/env python3
"""Run the independent reference skill; R regrasp, H hold, X reset simulation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import SimpleQueue
import time

import mujoco

from twin_sim.regrasp_microskill import MicroSkillConfig, RegraspMicroSkill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view', action='store_true')
    parser.add_argument('--auto', action='store_true', help='automatically request three successive regrasp actions')
    parser.add_argument('--randomize-shape', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=Path, default=Path('local/outputs/potato/regrasp_microskill'))
    parser.add_argument('--video', action='store_true', help='record headless run (set MUJOCO_GL=egl)')
    args = parser.parse_args()
    if args.video and args.view:
        parser.error('--video is for headless execution')
    config = MicroSkillConfig(seed=args.seed, randomize_shape=args.randomize_shape)
    skill = RegraspMicroSkill(config)
    args.output.mkdir(parents=True, exist_ok=True)
    records, requests = [], []
    queue = SimpleQueue()
    last_keys = {}

    def key_callback(key):
        now = time.monotonic()
        previous = last_keys.get(key, -10)
        last_keys[key] = now
        # Viewer only exposes key-down callbacks: suppress typematic repeats.
        # A new press needs a 0.6 s quiet interval, not a true gamepad edge reader.
        if now-previous > .6:
            queue.put(key)

    def tick(auto):
        while not queue.empty():
            key = queue.get()
            if key == ord('R'):
                result = skill.request(f'key-{len(requests)}')
                requests.append({'time': skill.data.time, 'result': result})
                print('request:', result, flush=True)
            elif key == ord('H'):
                skill.cancel()
            elif key == ord('X'):
                skill.reset()
        if auto and skill.phase in ('READY', 'HOLD') and skill.completed < 3:
            skill.request(f'auto-{skill.completed}')
        status = skill.step()
        records.append(status)
        return status

    if args.view:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(skill.model, skill.data, key_callback=key_callback) as viewer:
            viewer.cam.lookat[:] = [.43, 0, .29]
            viewer.cam.distance = .48
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            print('R: regrasp once | H: hold/cancel | X: reset simulation', flush=True)
            while viewer.is_running():
                start = time.monotonic()
                with viewer.lock():
                    status = tick(args.auto)
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   'REFERENCE / NO RL\n'+status['phase'],
                                   f"Completed: {status['completed']}\nDrift: {status['max_drift_mm']:.2f} mm\nR regrasp | H hold | X reset")])
                viewer.sync()
                time.sleep(max(0, skill.dt-(time.monotonic()-start)))
    else:
        renderer = writer = None
        if args.video:
            import imageio.v2 as imageio
            skill.model.vis.global_.offwidth = 800
            skill.model.vis.global_.offheight = 608
            renderer = mujoco.Renderer(skill.model, height=608, width=800)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [.43, 0, .29]
            camera.distance, camera.azimuth, camera.elevation = .48, 135, -25
            writer = imageio.get_writer(args.output/'reference.mp4', fps=25)
        try:
            for i in range(2000):
                status = tick(True)
                if writer is not None and i % 2 == 0:
                    renderer.update_scene(skill.data, camera=camera)
                    writer.append_data(renderer.render())
                if status['phase'] == 'FAULT' or skill.completed == 3:
                    break
        finally:
            if writer is not None:
                writer.close()
                renderer.close()
    report = {'controller': 'reference, not RL', 'config': asdict(config),
              'shape': asdict(skill.shape), 'result': skill.status(), 'events': skill.events,
              'requests': requests, 'steps': records}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(skill.status(), indent=2))
    if not args.view and skill.completed != 3:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
