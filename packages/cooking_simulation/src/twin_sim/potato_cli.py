"""Run a recorded baseline, train PPO, or evaluate and film a saved policy."""

import argparse
import json
from pathlib import Path

import numpy as np

from robot_core.paths import local_root
from twin_sim.potato_env import PotatoRegraspEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay", "train", "evaluate"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=local_root()/"outputs/potato")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--no-tactile", action="store_true")
    parser.add_argument("--no-disturbance", action="store_true")
    args = parser.parse_args(argv)
    if args.steps <= 0 or args.episodes <= 0:
        parser.error("steps and episodes must be positive")
    if args.mode == "evaluate" and args.policy is None:
        parser.error("evaluate requires --policy")
    args.output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(reference_path=args.reference, disturbance=not args.no_disturbance,
                  tactile_enabled=not args.no_tactile)
    if args.mode == "train":
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        torch.set_num_threads(1)
        with PotatoRegraspEnv(**kwargs) as env:
            model = PPO("MultiInputPolicy", Monitor(env), seed=args.seed, device="cpu",
                        n_steps=256, batch_size=64, n_epochs=5, verbose=1,
                        policy_kwargs={"net_arch": [128, 128]})
            model.learn(total_timesteps=args.steps)
            model.save(args.output/"ppo_potato")
            (args.output/"training.json").write_text(json.dumps({
                "seed": args.seed, "requested_steps": args.steps,
                "actual_steps": model.num_timesteps, "reference": str(env.reference_path),
                "tactile_enabled": not args.no_tactile, "disturbance": not args.no_disturbance,
                "privileged_actor": True, "knife_mode": "parked",
                "note": "Training completion does not establish task success. Evaluate on held-out seeds."
            }, indent=2)+"\n")
        return 0
    policy = None
    if args.policy:
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        policy = PPO.load(args.policy, device="cpu")
    summaries = []
    with PotatoRegraspEnv(**kwargs, render_mode="rgb_array" if args.video else None) as env:
        for episode in range(args.episodes):
            seed = args.seed+episode
            observation, info = env.reset(seed=seed)
            rows, tactile, positions = [], [], []
            writer = None
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(args.output/f"{args.mode}-{seed}.mp4"), fps=25)
            try:
                total_reward = 0.0
                for step in range(700):
                    action = np.zeros(28, dtype=np.float32) if policy is None else policy.predict(observation, deterministic=True)[0]
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    rows.append([env.data.time, info["displacement_m"], info["potato_hand_force_n"], info["potato_penetration_m"], info["reference_phase"]])
                    tactile.append(observation["tactile"].copy())
                    positions.append(env.data.qpos.copy())
                    if writer is not None and step % 4 == 0:
                        from PIL import Image, ImageDraw
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 48), fill=(15, 20, 28))
                        draw.text((8, 5), f"{args.mode} | rigid potato | knife PARKED | privileged state", fill="white")
                        draw.text((8, 25), f"t={env.data.time:.2f}s  drift={info['displacement_m']*1000:.1f}mm  hand force={info['potato_hand_force_n']:.2f}N", fill="white")
                        writer.append_data(np.asarray(frame))
                    if terminated or truncated:
                        break
            finally:
                if writer is not None:
                    writer.close()
            rows = np.asarray(rows)
            np.savez_compressed(args.output/f"{args.mode}-{seed}.npz", metrics=rows,
                                metric_names=np.array(["time_s", "displacement_m", "hand_force_n", "penetration_m", "phase"]),
                                tactile=np.asarray(tactile), qpos=np.asarray(positions))
            summaries.append({"seed": seed, "steps": step+1, "reward": total_reward,
                              "max_displacement_m": float(rows[:, 1].max()),
                              "max_hand_force_n": float(rows[:, 2].max()), **info})
    report = {"mode": args.mode, "knife_mode": "parked", "privileged_actor": True,
              "tactile_enabled": not args.no_tactile, "disturbance": not args.no_disturbance,
              "episodes": summaries, "success_rate": float(np.mean([r["is_success"] for r in summaries]))}
    (args.output/f"{args.mode}-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
