
# region Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
# 忽略來自 pygame 的關於 pkg_resources 的特定警告
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')

import os
import cv2
import time
import torch
import random
# import pygame
import imageio
import argparse
import numpy as np
from functools import partial
from torch.utils.tensorboard import SummaryWriter


# from smac.env import StarCraft2Env
# from pettingzoo.mpe import simple_spread_v3

# from storage import Storage
# from logger import Logger
from config import get_config
from utils import make_eval_env, state_preprocess, take_action_deterministic
# from FMANet import FMANetwork
from FMANet import FMANetwork
# endregion Imports

args = get_config()

base_seed = args.seed
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
# torch.autograd.set_detect_anomaly(True)


def action_to_env(action, action_dims, args):
    """把 policy sample 出來的 action index 轉成環境需要的 one-hot action。"""
    if not args.a_is_multi_discrete:
        action_np = np.squeeze(action, axis=-1) if action.ndim == 3 else action
        return np.eye(action_dims[0])[action_np]

    action_env = None
    for i in range(len(action_dims)):
        branch_one_hot = np.eye(action_dims[i])[action[:, :, i]]
        action_env = branch_one_hot if action_env is None else np.concatenate((action_env, branch_one_hot), axis=2)
    return action_env


@torch.no_grad()
def experiment_eval(args):
    print("==============Starting Evaluation Program...")
    cuda_is_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_is_available else "cpu")
    args.device = device
    if cuda_is_available:
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(base_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Using CPU device")

    # print("\n========== all_args Parameters ==========")
    # for k, v in vars(args).items():
    #     print(f"{k}: {v}")
    # print("======================================\n")

    env = make_eval_env(args)
    obs_batch = env.reset()
    # print("obs_batch shape:", obs_batch.shape)   # (num_envs, num_agents, obs_dim)
    obs_dim = obs_batch.shape[-1]       # (num_envs, num_agents, obs_dim)
    state_tensor, obs_tensor = state_preprocess(obs_batch, obs_dim, args, device)
    
    if args.env_name == "simple_spread_v3":
        args.a_is_multi_discrete = False
        action_dims = [5]
    elif args.env_name == "simple_reference_v3":
        args.a_is_multi_discrete = True
        action_dims = [5, 10]

    print("action_space =", env.action_space)
    print("Action dims:", action_dims)
    print("Using Seed: ", base_seed)

    Net = FMANetwork(
        num_envs=args.num_envs,   # 16
        num_agents=args.num_agents,     # 3
        input_dim_G=obs_dim*(args.num_agents) if args.use_attention else obs_dim*(args.num_agents+1),
        input_dim_L=obs_dim,  
        hidden_dim_manager=args.hidden_dim_manager,  # 256
        hidden_dim_worker=args.hidden_dim_worker,  # 16
        n_actions=action_dims,  # 18
        time_horizon=args.time_horizon,  # 10
        dilation=args.dilation,  # 10
        device=device,
        args=args)
    
    Net.load_state_dict(torch.load(f'models/{args.env_name}/{args.load_model}.pt', map_location=device))
    Net.eval()

    goals, states, masks = Net.init_obj()
    global_steps = 0
    info = {}
    info['returns/episodic_reward'] = np.empty(args.num_envs, dtype=object)
    info['returns/episodic_length'] = np.empty(args.num_envs, dtype=int)
    info['returns/final_reward'] = np.empty(args.num_envs, dtype=float)
    total_rewards = np.zeros(args.num_envs)
    steps = np.zeros(args.num_envs)
    episode = 0

    out_filename = f"records/{args.env_name}/{args.load_model}_seed{base_seed}"
    all_frames = []
    calc_start = time.time()
    r = env.render()
    image = env.render('rgb_array')[0][0]
    cv2.putText(image, f"Step: 0", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(image, f"Reward: 0.000", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    all_frames.append(image)
    
    while episode < args.test_episodes*args.num_envs:
        with torch.no_grad():
            global_steps += 1
            Net.repackage_hidden()
            goals = [g.detach() for g in goals]

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
            
            action_dist, goals, states, _, _, _ = Net(state_tensor, obs_tensor, goals, states, masks[-1], save=True)

            if args.a_is_multi_discrete:
                actions = []
                for probs in action_dist:
                    action_branch = torch.argmax(probs, dim=-1)  # (E, A)
                    actions.append(action_branch)

                action_t = torch.stack(actions, dim=-1)  # (E, A, branches)

            else:
                action_t = torch.argmax(action_dist, dim=-1, keepdim=True)  # (E, A, 1)

            action_np = action_t.detach().cpu().numpy()
            action_env = action_to_env(action_np, action_dims, args)


            # action, logp, entropy = take_action_deterministic(action_dist)
            # action_env = np.eye(action_dims[0])[action]
            obs_batch, rewards, done, infos = env.step(action_env) # shape : (envs*agents, )

            state_tensor, obs_tensor = state_preprocess(obs_batch, obs_dim, args, device)
            rewards = rewards.reshape(args.num_envs, args.num_agents)   # (envs, agents)
            reward = np.max(rewards, axis=1)    # (envs, )
            done_tensor = torch.tensor(done, dtype=torch.float32, device=device).reshape(args.num_envs*args.num_agents, 1)      # (envs, agents)

            total_rewards += reward
            steps += 1
            mask = 1.0 - done_tensor   # (envs*agents, 1)

            masks.pop(0)
            masks.append(mask.detach())
            global_steps += args.num_envs

            image = env.render('rgb_array')[0][0]
            cv2.putText(image, f"Step: {int(steps[0])}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, f"Reward: {reward[0]:.3f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(image, f"episodic_Reward: {total_rewards[0]:.3f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            all_frames.append(image)

            calc_end = time.time()
            elapsed = calc_end - calc_start
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)
                
            for i in range(done.shape[0]):
                if done[i][0]:
                    episode += 1
                    print("Episode ", episode, " finished. Total Reward: ", total_rewards[i], "Final Reward: ", rewards[i][0], " Steps: ", steps[i])
                    total_rewards[i] = 0
                    steps[i] = 0

    print(f"Video saved to {out_filename}")
    # imageio.mimsave(f'{out_filename}_ep{episode // args.num_envs}.gif', all_frames, duration=0.1)
    imageio.mimsave(f'{out_filename}.gif', all_frames, duration=0.1)
    all_frames = []
    calc_start = time.time()
                        



def main(args):
    run_name = args.run_name
    args.run_name = f"{run_name}"
    args.num_envs = 1
    args.evaluate = True
    args.render = "rgb_array"
    if args.env_name == "simple_spread_v3":
        args.scenario_name = 'simple_spread'
        args.landmarks = 3
        args.num_agents = 3
    elif args.env_name == "simple_reference_v3":
        args.scenario_name = 'simple_reference'
        args.landmarks = 3
        args.num_agents = 2
    else:
        print("Environment not recognized in train.py")
    experiment_eval(args)


if __name__ == '__main__':
    main(args)
