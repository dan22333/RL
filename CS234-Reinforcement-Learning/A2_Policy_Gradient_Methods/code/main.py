# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import torch
import gym
import pybullet_envs
from policy_gradient import PolicyGradient
from ppo import PPO
from config import get_config
import random

import pdb

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name", required=True, type=str, choices=["cartpole", "pendulum", "cheetah"]
)
parser.add_argument("--baseline", dest="use_baseline", action="store_true")
parser.add_argument("--no-baseline", dest="use_baseline", action="store_false")
parser.add_argument("--ppo", dest="ppo", action="store_true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--render", dest="render", action="store_true",
    help="Open a live PyBullet GUI and replay the policy while training.",
)
parser.add_argument(
    "--render-freq", dest="render_freq", type=int, default=10,
    help="Render a GUI episode every N training iterations (with --render).",
)
parser.add_argument(
    "--render-live", dest="render_live", action="store_true",
    help="Show the actual training rollouts continuously in one GUI window "
         "(no pausing/freezing); watch the agent improve in real time.",
)
parser.add_argument(
    "--render-fps", dest="render_fps", type=int, default=0,
    help="Cap live-render speed to N fps (0 = as fast as possible). Used with --render-live.",
)
# GUI camera controls (side view by default); tweak without editing code
parser.add_argument(
    "--record-video", dest="record_video", action="store_true",
    help="Save offscreen GIF videos of the policy every --render-freq iters (no display needed).",
)
parser.add_argument("--render-yaw", dest="render_yaw", type=float, default=0.0)
parser.add_argument("--render-pitch", dest="render_pitch", type=float, default=-5.0)
parser.add_argument("--render-dist", dest="render_dist", type=float, default=2.5)

parser.set_defaults(use_baseline=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config = get_config(args.env_name, args.use_baseline, args.ppo, args.seed)
    config.render = args.render
    config.record_video = args.record_video
    config.render_freq = args.render_freq
    config.render_live = args.render_live
    config.render_fps = args.render_fps
    config.render_yaw = args.render_yaw
    config.render_pitch = args.render_pitch
    config.render_dist = args.render_dist
    env = gym.make(config.env_name)
    # train model
    model = PolicyGradient(env, config, args.seed) if not args.ppo else PPO(env, config, args.seed)
    model.run()
