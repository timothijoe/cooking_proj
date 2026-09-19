#!/usr/bin/env python3
"""Train/evaluate actual PPO residual control in the independent no-knife task."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from twin_sim.regrasp_rl import RegraspRLEnv


def evaluate(policy, seeds):
    results = []
    for seed in seeds:
        env = RegraspRLEnv()
        obs, _ = env.reset(seed=seed, options={'shape_seed': seed})
        total, actions = 0., []
        while True:
            action = np.zeros(8, dtype=np.float32) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            actions.append(action.tolist())
            if terminated or truncated:
                break
        results.append({'seed': seed, 'return': total, 'result': info,
                        'mean_abs_action': np.abs(actions).mean(axis=0).tolist(),
                        'events': env.skill.events})
        print('evaluation', 'reference' if policy is None else 'PPO', seed,
              info['completed'], info['reason'], info['max_drift_mm'], flush=True)
        env.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=16384)
    p.add_argument('--output', type=Path, default=Path('local/outputs/potato/regrasp_rl_v1'))
    p.add_argument('--load', type=Path)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[100, 101, 102, 103])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = Monitor(RegraspRLEnv(), filename=str(args.output/'training'))
    policy = PPO.load(args.load, env=env, device='cpu') if args.load else PPO(
        'MlpPolicy', env, device='cpu', seed=0, n_steps=512, batch_size=128,
        learning_rate=3e-4, gamma=.995, verbose=1,
        policy_kwargs={'net_arch': [64, 64], 'log_std_init': -1.5})
    start = time.monotonic()
    if args.steps:
        policy.learn(total_timesteps=args.steps, reset_num_timesteps=not bool(args.load))
        policy.save(args.output/'policy')
    metadata = {'requested_steps': args.steps, 'actual_total_steps': policy.num_timesteps,
                'training_seconds': time.monotonic()-start,
                'algorithm': 'PPO residual control, retained reference phases/IK',
                'observation': '354 floats: three frames of proprioception, ideal tactile, torque proxy, phase, previous action; no object truth',
                'action': '8: three load targets, approach speed, support speed, flight speed, flight lift, DIP curl',
                'train_shape_seeds': 'RNG seed 0; sampled per episode',
                'evaluation_shape_seeds': args.eval_seeds}
    (args.output/'training.json').write_text(json.dumps(metadata, indent=2)+'\n')
    reference = evaluate(None, args.eval_seeds)
    learned = evaluate(policy, args.eval_seeds)
    (args.output/'evaluation.json').write_text(json.dumps({'metadata': metadata, 'reference': reference, 'ppo': learned}, indent=2)+'\n')
    env.close()


if __name__ == '__main__':
    main()
