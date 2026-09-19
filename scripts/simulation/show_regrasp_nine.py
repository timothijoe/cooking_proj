#!/usr/bin/env python3
"""Record/play three physical rollouts: nine wrist advances, three fingertip relocations.

The three-part schedule is programmed; PPO weights are reused without retraining.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from twin_sim.regrasp_rl import RegraspRLEnv


def record(policy, seed, output):
    env = RegraspRLEnv(support_repeats=3)
    obs, _ = env.reset(seed=seed, options={'shape_seed': seed})
    skill = env.skill
    skill.model.vis.global_.offwidth, skill.model.vis.global_.offheight = 960, 720
    renderer = mujoco.Renderer(skill.model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [.43, 0, .30]
    camera.distance, camera.azimuth, camera.elevation = .43, 135, -22
    video = output/f'seed_{seed}.mp4'
    writer = imageio.get_writer(video, fps=25)
    records = []

    def frame(label=None):
        renderer.update_scene(skill.data, camera=camera)
        picture = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 960, 92), fill=(18, 24, 30))
        phase = label or skill.phase
        draw.text((16, 10), 'PPO residual + SCRIPTED 3 x 3 schedule (not retrained)', fill='white', font_size=20)
        draw.text((16, 38), f'Seed {seed} | {phase}', fill='white', font_size=18)
        draw.text((16, 65), f'Wrist advances: {skill.wrist_retreats}/9    Fingertip relocations: {skill.completed}/3    Drift: {skill.max_drift*1000:.2f} mm', fill='white', font_size=18)
        return np.asarray(picture)

    try:
        initial = frame('START ABOVE POTATO - preview hold')
        for _ in range(40):
            writer.append_data(initial)
        for step in range(env.max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            records.append({**info, 'wrist_xyz': skill.data.site_xpos[skill.model.site('left_palm_tcp_site').id].tolist(),
                            'tips_xyz': [skill.data.geom_xpos[f.tip].tolist() for f in skill.fingers],
                            'action': action.tolist()})
            if step % 2 == 0 or term or trunc:
                writer.append_data(frame())
            if term or trunc:
                break
        result = skill.status()
        for step in range(100):
            skill.step()
            if step % 2 == 0:
                writer.append_data(frame('COMPLETE - HOLD' if result['completed'] == 3 else 'FAILED - '+result['reason']))
    finally:
        writer.close()
        renderer.close()
    report = {'seed': seed, 'controller': 'saved PPO residual, new scripted 3x3 schedule, not retrained',
              'wrist_total_support_retreat_per_round_mm': 4,
              'wrist_retreat_per_segment_mm': 4/3,
              'fingertip_retreat_per_round_mm': 8,
              'result': result, 'events': skill.events, 'steps': records}
    (output/f'seed_{seed}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(seed, result, flush=True)
    if result['completed'] != 3 or result['wrist_retreats'] != 9:
        raise RuntimeError(f'Failed 3x3 rollout: seed {seed}: {result}')
    return video


def play(videos):
    import tkinter as tk
    from PIL import ImageTk
    root = tk.Tk()
    root.title('倒手演示：九次退腕 / 三次移指（物理仿真回放）')
    label = tk.Label(root)
    label.pack()
    controls = tk.Frame(root)
    controls.pack(fill='x')
    status = tk.StringVar()
    tk.Label(root, textvariable=status).pack()
    index, reader, count, paused = 0, None, 0, False

    def switch(delta=0):
        nonlocal index, reader, count
        if reader is not None:
            reader.close()
        index = (index+delta) % len(videos)
        reader = imageio.get_reader(videos[index])
        count = 0

    def pause():
        nonlocal paused
        paused = not paused

    tk.Button(controls, text='上一个', command=lambda: switch(-1)).pack(side='left')
    tk.Button(controls, text='暂停 / 继续（空格）', command=pause).pack(side='left')
    tk.Button(controls, text='重播', command=switch).pack(side='left')
    tk.Button(controls, text='下一个', command=lambda: switch(1)).pack(side='left')
    root.bind('<space>', lambda _: pause())
    root.bind('<Left>', lambda _: switch(-1))
    root.bind('<Right>', lambda _: switch(1))
    switch()

    def tick():
        nonlocal count
        if not paused:
            try:
                picture = reader.get_data(count)
            except IndexError:
                switch(1)
                picture = reader.get_data(0)
            image = ImageTk.PhotoImage(Image.fromarray(picture))
            label.configure(image=image)
            label.image = image
            count += 1
        status.set(f'第 {index+1}/{len(videos)} 段 | {videos[index].name} | 程序编排三次退腕后移指，PPO 调整参数 | '+('已暂停' if paused else '播放中'))
        root.after(40, tick)

    root.protocol('WM_DELETE_WINDOW', root.destroy)
    root.after(0, tick)
    root.lift()
    root.attributes('-topmost', True)
    root.after(1500, lambda: root.attributes('-topmost', False))
    try:
        root.mainloop()
    finally:
        if reader is not None:
            reader.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, default=Path('local/outputs/potato/regrasp_rl_v1/policy.zip'))
    p.add_argument('--output', type=Path, default=Path('local/outputs/potato/regrasp_nine'))
    p.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102])
    p.add_argument('--play', action='store_true', help='play existing recorded videos; no new simulation')
    args = p.parse_args()
    if args.play:
        play([args.output/f'seed_{s}.mp4' for s in args.seeds])
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        policy = PPO.load(args.policy, device='cpu')
        for seed in args.seeds:
            record(policy, seed, args.output)


if __name__ == '__main__':
    main()
