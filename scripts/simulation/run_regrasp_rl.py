#!/usr/bin/env python3
"""Execute a saved PPO policy, visibly labelled; no training in this runner."""
import argparse
import json
from pathlib import Path
import time

import mujoco
from stable_baselines3 import PPO
from twin_sim.regrasp_rl import RegraspRLEnv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--view', action='store_true')
    p.add_argument('--video', action='store_true')
    p.add_argument('--output', type=Path, default=Path('local/outputs/potato/regrasp_rl_v1/rollout'))
    args = p.parse_args()
    if args.video and args.view:
        p.error('choose --view or --video')
    policy = PPO.load(args.policy, device='cpu')
    env = RegraspRLEnv()
    obs, _ = env.reset(seed=args.seed, options={'shape_seed': args.seed})
    skill = env.skill
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    done = False

    def tick():
        nonlocal obs, done
        if done:
            skill.step()  # Keep physical integration alive after completion.
            return
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        records.append({'action': action.tolist(), **info})
        done = term or trunc

    def camera(cam):
        cam.lookat[:] = [.43, 0, .29]
        cam.distance, cam.azimuth, cam.elevation = .48, 135, -25

    if args.view:
        from mujoco import viewer as mjviewer
        with mjviewer.launch_passive(skill.model, skill.data) as viewer:
            camera(viewer.cam)
            while viewer.is_running():
                start = time.monotonic()
                with viewer.lock():
                    tick()
                status = skill.status()
                viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   'TRAINED PPO RESIDUAL\n'+status['phase'],
                                   f"Seed {args.seed} | {status['completed']}/3\nDrift {status['max_drift_mm']:.2f} mm\nReference phases + learned adjustments")])
                viewer.sync()
                time.sleep(max(0, skill.dt-(time.monotonic()-start)))
    else:
        renderer = writer = None
        if args.video:
            import imageio.v2 as imageio
            skill.model.vis.global_.offwidth, skill.model.vis.global_.offheight = 800, 608
            renderer = mujoco.Renderer(skill.model, height=608, width=800)
            cam = mujoco.MjvCamera()
            camera(cam)
            writer = imageio.get_writer(args.output/'ppo.mp4', fps=25)
        try:
            for i in range(env.max_steps):
                tick()
                if writer is not None and (i % 2 == 0 or done):
                    renderer.update_scene(skill.data, camera=cam)
                    from PIL import Image, ImageDraw
                    import numpy as np
                    frame = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 800, 54), fill=(20, 25, 30))
                    draw.text((12, 8), 'TRAINED PPO RESIDUAL | '+skill.phase, fill='white')
                    draw.text((12, 30), f'Seed {args.seed} | completed {skill.completed}/3 | reference phases + learned adjustments', fill='white')
                    writer.append_data(np.asarray(frame))
                if done:
                    break
        finally:
            if writer is not None:
                writer.close()
                renderer.close()
    report = {'policy': str(args.policy.resolve()), 'shape_seed': args.seed,
              'controller': 'trained PPO residual', 'result': skill.status(), 'steps': records}
    (args.output/'rollout.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['result'], indent=2))
    env.close()


if __name__ == '__main__':
    main()
