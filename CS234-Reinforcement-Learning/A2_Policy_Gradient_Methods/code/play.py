# -*- coding: UTF-8 -*-
"""Replay a trained policy saved by training (see PolicyGradient.save_model).

Loads the policy weights from a run's `model.weights/policy.pt`, rebuilds the same
architecture from the config, and runs a few greedy-ish episodes — optionally in a
live PyBullet GUI window (run this from YOUR OWN terminal for the window to appear).

Examples:
    # just print rewards, no window
    python play.py --env-name cartpole --ppo --episodes 5
    # watch it live from the side, real-time
    python play.py --env-name cheetah --ppo --render --render-yaw 180 --render-fps 60
"""
import argparse
import os
import time

import gym
import numpy as np
import pybullet
import pybullet_envs  # noqa: F401  (registers the Bullet envs)
import torch

from config import get_config
from policy_gradient import PolicyGradient
from ppo import PPO

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", required=True, choices=["cartpole", "pendulum", "cheetah"])
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true")
parser.add_argument("--seed", type=int, default=1, help="which trained seed to load")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--render", action="store_true", help="open a live GUI window")
parser.add_argument("--render-fps", type=int, default=60, help="0 = as fast as possible")
parser.add_argument("--render-yaw", type=float, default=180.0)
parser.add_argument("--render-pitch", type=float, default=-10.0)
parser.add_argument("--render-dist", type=float, default=3.0)
parser.set_defaults(use_baseline=True)


def main():
    args = parser.parse_args()
    config = get_config(args.env_name, args.use_baseline, args.ppo, args.seed)

    weights = os.path.join(config.model_output, "policy.pt")
    if not os.path.exists(weights):
        raise SystemExit(
            "No saved model at {} -- train it first (main.py ... same flags/seed).".format(weights)
        )

    env = gym.make(config.env_name)
    env.seed(args.seed)
    model = PPO(env, config, args.seed) if args.ppo else PolicyGradient(env, config, args.seed)
    model.policy.load_state_dict(torch.load(weights, map_location="cpu"))
    model.policy.eval()
    print("Loaded {}".format(weights))

    if args.render:
        env.render(mode="human")  # must precede first reset for a GUI window
        try:
            p = env.unwrapped._p
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        except Exception:
            pass

    def aim_camera():
        try:
            u = env.unwrapped
            p = u._p
            try:
                x, y, z = u.robot.body_xyz
            except Exception:
                x, y, z = p.getBasePositionAndOrientation(u.cartpole)[0]
            p.resetDebugVisualizerCamera(args.render_dist, args.render_yaw, args.render_pitch, [x, y, z])
        except Exception:
            pass

    returns = []
    for ep in range(args.episodes):
        try:
            state = env.reset()
        except pybullet.error:
            break
        total = 0.0
        for _ in range(config.max_ep_len):
            action = model.policy.act(state[None])[0]
            try:
                state, reward, done, _ = env.step(action)
            except pybullet.error:
                done = True
                reward = 0.0
            total += reward
            if args.render:
                aim_camera()
                if args.render_fps:
                    time.sleep(1.0 / args.render_fps)
            if done:
                break
        returns.append(total)
        print("episode {}: reward {:.1f}".format(ep + 1, total))

    if returns:
        print("mean reward over {} episodes: {:.1f} +/- {:.1f}".format(
            len(returns), float(np.mean(returns)), float(np.std(returns))))


if __name__ == "__main__":
    main()
