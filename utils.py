"""
This file is filled with miscelaneous classes and functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
from torch.distributions import Categorical
import time
import numpy as np
from typing import Optional
# import gymnasium as gym
# from gymnasium import spaces
# from gymnasium.core import Wrapper
# from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.mpe.MPE_env import MPEEnv
# from smac.env import StarCraft2Env
# from pettingzoo.mpe import simple_spread_v3, simple_reference_v3, simple_push_v3


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.num_envs == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.num_envs)])
    
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = MPEEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    return DummyVecEnv([get_env_fn(0)])

def state_preprocess(obs_batch, obs_dim, args, device, train=True):
    if train:
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).reshape(args.num_envs, args.num_agents, obs_dim).to(device)  # (envs, agents, obs_dim)
    else:
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).reshape(args.num_envs, args.num_agents, obs_dim).to(device)  # (envs, agents, obs_dim)
        
    state = obs_tensor.reshape(obs_tensor.shape[0], -1)  # (envs, agents*obs_dim)
    if args.use_attention:
        state_tensor = state        # (envs, agents*obs_dim)
    else:
        state_rep = state.unsqueeze(1).repeat(1, args.num_agents, 1)       # (envs, agents, agents*obs_dim)
        state_tensor = torch.cat([state_rep, obs_tensor], dim=2)      # (envs, agents, (agents+1)*obs_dim)
    return state_tensor, obs_tensor


# def take_action(a):
#     """
#     Returns:
#         action:
#             if Discrete -> (envs, agents)
#             if MultiDiscrete -> (envs, agents, num_branches)
#         logp:
#             (envs, agents)
#         entropy:
#             (envs, agents)
#     """
#     if not a.__class__.__name__ == "MultiDiscrete":
#         dist = a
#         action = dist.sample()                         # (envs, agents)
#         logp = dist.log_prob(action)                   # (envs, agents)
#         entropy = dist.entropy()                       # (envs, agents)
#         return action.cpu().numpy(), logp, entropy

#     else:
#         # action_dist is a list: [dist_move, dist_comm]
#         dists = a

#         actions = []
#         logps = []
#         entropies = []

#         for dist in dists:
#             a = dist.sample()                         # (envs, agents)
#             lp = dist.log_prob(a)                     # (envs, agents)
#             ent = dist.entropy()                      # (envs, agents)

#             actions.append(a)
#             logps.append(lp)
#             entropies.append(ent)

#         # (num_branch, envs, agents) -> (envs, agents, num_branch)
#         action = torch.stack(actions, dim=-1)
#         logp = sum(logps)         # joint log prob
#         entropy = sum(entropies)  # joint entropy

#         return action.cpu().numpy(), logp, entropy


def take_action(action_dist):
    # Discrete
    if not isinstance(action_dist, list):
        probs = action_dist
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        if action.dim() == 3 and action.size(-1) == 1:
            action = action.squeeze(-1)
            logp = logp.squeeze(-1)
            entropy = entropy.squeeze(-1)

        return action.cpu().numpy(), logp, entropy

    # MultiDiscrete
    else:
        actions = []
        logps = []
        entropies = []

        for probs in action_dist:
            dist = torch.distributions.Categorical(probs=probs)
            a = dist.sample()
            lp = dist.log_prob(a)
            ent = dist.entropy()

            actions.append(a)
            logps.append(lp)
            entropies.append(ent)

        action = torch.stack(actions, dim=-1)  # (envs, agents, num_branches)
        logp = sum(logps)                      # joint log prob
        entropy = sum(entropies)              # joint entropy

        return action.cpu().numpy(), logp, entropy


# def take_action_deterministic(a):
#     dist = Categorical(a)  # distribution of action
#     action = torch.argmax(a, dim=-1)
#     logp = dist.log_prob(action)
#     entropy = dist.entropy()
#     return action.cpu().detach().numpy(), logp, entropy


def take_action_deterministic(action_dist):
    # Discrete
    if not isinstance(action_dist, list):
        probs = action_dist
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.argmax(probs, dim=-1)
        logp = dist.log_prob(action)
        entropy = dist.entropy()

        if action.dim() == 3 and action.size(-1) == 1:
            action = action.squeeze(-1)
            logp = logp.squeeze(-1)
            entropy = entropy.squeeze(-1)

        return action.cpu().numpy(), logp, entropy

    # MultiDiscrete
    else:
        actions = []
        logps = []
        entropies = []

        for probs in action_dist:
            dist = torch.distributions.Categorical(probs=probs)
            a = torch.argmax(probs, dim=-1)
            lp = dist.log_prob(a)
            ent = dist.entropy()

            actions.append(a)
            logps.append(lp)
            entropies.append(ent)

        action = torch.stack(actions, dim=-1)  # (envs, agents, num_branches)
        logp = sum(logps)                      # joint log prob
        entropy = sum(entropies)              # joint entropy

        return action.cpu().numpy(), logp, entropy


def take_action_masked(a, avail_actions):
    epsilon = 1e-8
    if not torch.is_tensor(avail_actions):
        mask = torch.tensor(avail_actions, device=a.device, dtype=a.dtype)
    else:
        mask = avail_actions.to(a.device).to(a.dtype)
    masked_a = a * mask  # mask the action
    masked_a = masked_a / (masked_a.sum(dim=-1, keepdim=True)+epsilon)      # normalize the action
    dist = Categorical(masked_a)  # distribution of action
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy

def init_hidden(n_workers, h_dim, device, grad=False):
    return (torch.zeros(n_workers, h_dim, requires_grad=grad).to(device),
            torch.zeros(n_workers, h_dim, requires_grad=grad).to(device))


def weight_init(layer):
    if type(layer) == torch.nn.modules.conv.Conv2d or \
            type(layer) == torch.nn.Linear:
        torch.nn.init.orthogonal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, 0)
