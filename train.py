# region Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pygame.pkgdata')

import os
import torch
import random
# import pygame
import numpy as np
from tqdm import tqdm
from collections import deque

from logger import Logger
from config import get_config
from utils import make_train_env, state_preprocess
from FMANet import FMANetwork, FG_loss
# endregion Imports

os.environ["SDL_VIDEODRIVER"] = "dummy"  # 為了在無顯示器的環境下使用 pygame
args = get_config()
base_seed = args.seed
random.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
# torch.autograd.set_detect_anomaly(True)

# region Sub Finctions
def clone_tensor_list(xs):
    """複製 goal/state/mask history，避免 PPO update 重放時改到真實 rollout 狀態。"""
    return [x.detach().clone() for x in xs]


def clone_hidden(hidden):
    """複製 Net.hidden_m / Net.hidden_w。hidden 通常是 [h, c] 或 (h, c)。"""
    return [h.detach().clone() for h in hidden]


def detach_hidden(hidden):
    return [h.detach() for h in hidden]


def policy_logp_entropy(action_dist, action, args, eps=1e-8):
    """
    根據已採樣 action 重新計算 log_prob / entropy。

    為什麼要自己算：
    - PPO/MAPPO 需要 old_logp 與 new_logp。
    - update 階段不能重新 sample action，而是要對 rollout 當時的 action 算 new_logp。
    - 你的 action_dist 是 softmax 後的機率，不是 torch.distributions 物件。
    """
    if args.a_is_multi_discrete:
        # action_dist: list，每個元素 shape = (num_envs, num_agents, dim)
        # action: shape = (num_envs, num_agents, num_action_branches)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.long, device=args.device)
        action = action.long().to(args.device)

        logp = 0.0
        entropy = 0.0
        for branch_id, probs in enumerate(action_dist):
            probs = probs.clamp(min=eps, max=1.0)
            branch_action = action[:, :, branch_id].unsqueeze(-1)  # (E, A, 1)
            branch_logp = torch.log(probs.gather(-1, branch_action))
            branch_entropy = -(probs * torch.log(probs)).sum(dim=-1, keepdim=True)
            logp = logp + branch_logp
            entropy = entropy + branch_entropy

        return logp.reshape(args.num_envs * args.num_agents, 1), entropy.reshape(args.num_envs * args.num_agents, 1)

    else:
        # action_dist: tensor shape = (num_envs, num_agents, action_dim)
        # action: shape = (num_envs, num_agents) or (num_envs, num_agents, 1)
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.long, device=args.device)
        action = action.long().to(args.device)
        if action.ndim == 2:
            action = action.unsqueeze(-1)
        elif action.ndim == 3 and action.shape[-1] != 1:
            action = action[..., :1]

        probs = action_dist.clamp(min=eps, max=1.0)
        logp = torch.log(probs.gather(-1, action))
        entropy = -(probs * torch.log(probs)).sum(dim=-1, keepdim=True)
        return logp.reshape(args.num_envs * args.num_agents, 1), entropy.reshape(args.num_envs * args.num_agents, 1)


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

# endregion Sub Finctions


def experiment(args):
    logger = Logger(args.run_name, args)
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

    print("\n========== all_args Parameters ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("======================================\n")

    print("Using Attention Mechanism" if args.use_attention else "Not using Attention Mechanism")

    env = make_train_env(args)
    obs_batch = env.reset()
    obs_dim = obs_batch.shape[-1]
    state_tensor, obs_tensor = state_preprocess(obs_batch, obs_dim, args, device, train=True)

    if args.env_name == "simple_spread_v3":
        args.a_is_multi_discrete = False
        action_dims = [5]
    elif args.env_name == "simple_reference_v3":
        args.a_is_multi_discrete = True
        action_dims = [5, 10]
    else:
        raise ValueError(f"Environment not recognized in train.py: {args.env_name}")

    print("action_space =", env.action_space)
    print("Action dims:", action_dims)

    Net = FMANetwork(
        num_envs=args.num_envs,
        num_agents=args.num_agents,
        input_dim_G=obs_dim * (args.num_agents) if args.use_attention else obs_dim * (args.num_agents + 1),
        input_dim_L=obs_dim,
        hidden_dim_manager=args.hidden_dim_manager,
        hidden_dim_worker=args.hidden_dim_worker,
        n_actions=action_dims,
        time_horizon=args.time_horizon,
        dilation=args.dilation,
        device=device,
        args=args
    )

    optimizer = torch.optim.RMSprop(Net.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)

    goals, states, masks = Net.init_obj()
    info = {
        'returns/episodic_reward': np.empty(args.num_envs, dtype=object),
        'returns/final_reward': np.empty(args.num_envs, dtype=float),
        'returns/moving_average_reward': np.empty(args.num_envs, dtype=float),
    }
    episodic_returns_history = deque(maxlen=args.window_size)
    total_rewards = np.zeros(args.num_envs)
    steps = np.zeros(args.num_envs)
    global_steps = 0
    episode = 1
    last_saved_episode = 0
    pbar = tqdm(total=args.max_total_steps, desc="Training", ncols=100)

    while global_steps < args.max_total_steps:
        # =========================================================
        # 1) Rollout collection
        # =========================================================
        # PPO/MAPPO 的重點：collection 階段只收資料，不建立要拿去多次 backward 的 graph。
        Net.train()
        Net.repackage_hidden()
        goals = [g.detach() for g in goals]
        states = [s.detach() for s in states]
        masks = [m.detach() for m in masks]

        rollout = {
            'state': [],
            'obs': [],
            'action': [],
            'r': [],
            'r_i': [],
            'm': [],
            'old_logp': [],
            'old_v_m': [],
            'old_v_w': [],

            # ===== Worker-only PPO replay 用 =====
            # 'z_L': [],
            # 'goal': [],
            # 'hidden_w': [],
            # 'mask_prev': [],

            # ===== replay sequence 初始狀態 =====
            'init_hidden_m': clone_hidden(Net.hidden_m),
            'init_hidden_w': clone_hidden(Net.hidden_w),
            'init_goals': clone_tensor_list(goals),
            'init_states': clone_tensor_list(states),
            'init_masks': clone_tensor_list(masks),
        }

        for _ in range(args.num_steps):
            # 重要：rollout 階段不需要梯度，避免同一張 graph 被多次 backward。
            with torch.no_grad():
                # hidden_w_before = clone_hidden(Net.hidden_w)
                # mask_prev = masks[-1].detach()
                action_dist, goals, states, value_m, value_w, attn_w = Net(
                    state_tensor, obs_tensor, goals, states, masks[-1], save=True
                )

                # ===== Worker replay 要用 =====
                # z_L 用 rollout 當下的 percept feature
                # z_L = Net.percept_L(obs_tensor).detach()


                # 用 helper 取 action，並另外用 policy_logp_entropy 算 old_logp，確保 update 時格式一致。
                # 這裡不用原本 take_action 回傳的 logp，是為了避免 logp 格式在 MultiDiscrete 時不一致。
                if args.a_is_multi_discrete:
                    sampled_actions = []
                    for probs in action_dist:
                        dist = torch.distributions.Categorical(probs=probs)
                        sampled_actions.append(dist.sample())  # (E, A)
                    action_t = torch.stack(sampled_actions, dim=-1)  # (E, A, branches)
                else:
                    dist = torch.distributions.Categorical(probs=action_dist)
                    action_t = dist.sample().unsqueeze(-1)  # (E, A, 1)

                old_logp, _ = policy_logp_entropy(action_dist, action_t, args)
                action_np = action_t.detach().cpu().numpy()
                action_env = action_to_env(action_np, action_dims, args)

            obs_batch, rewards, done, infos = env.step(action_env)

            next_state_tensor, next_obs_tensor = state_preprocess(obs_batch, obs_dim, args, device, train=True)
            rewards = rewards.reshape(args.num_envs, args.num_agents)
            reward = np.max(rewards, axis=1)

            done_tensor = torch.tensor(done, dtype=torch.float32, device=device).reshape(args.num_envs * args.num_agents, 1)
            mask = 1.0 - done_tensor

            # 更新真實 rollout 的 mask history，接下來 intrinsic/cosine 會用到。
            masks.pop(0)
            masks.append(mask.detach())

            # episode logging
            total_rewards += reward
            steps += 1
            global_steps += args.num_envs
            pbar.update(args.num_envs)

            if done[0][0]:
                for i in range(done.shape[0]):
                    episodic_returns_history.append(reward[i])
                moving_avg_return = np.mean(episodic_returns_history)
                info['returns/episodic_reward'] = np.mean(total_rewards)
                info['returns/final_reward'] = np.mean(reward)
                info['returns/moving_average_reward'] = moving_avg_return
                episode += args.num_envs
                total_rewards = np.zeros(args.num_envs)
                steps = np.zeros(args.num_envs)
                logger.log_scalars(info, global_steps)

            # 存 rollout 資料。全部 detach，update 階段會重新 forward。
            rollout['state'].append(state_tensor.detach())
            rollout['obs'].append(obs_tensor.detach())
            rollout['action'].append(action_t.detach())
            rollout['r'].append(torch.as_tensor(rewards, dtype=torch.float32, device=device).reshape(args.num_envs * args.num_agents, 1))
            rollout['r_i'].append(Net.intrinsic_reward(states, goals, masks).detach())
            rollout['m'].append(mask.detach())
            rollout['old_logp'].append(old_logp.detach())
            rollout['old_v_m'].append(value_m.detach())
            rollout['old_v_w'].append(value_w.detach())

            # ===== Worker-only PPO replay 用 =====
            # rollout['z_L'].append(z_L)
            # rollout['goal'].append(goals[-1].detach())
            # rollout['hidden_w'].append(hidden_w_before)
            # rollout['mask_prev'].append(mask_prev)

            state_tensor, obs_tensor = next_state_tensor, next_obs_tensor

        # 保存真實環境 rollout 結束後的 recurrent 狀態。
        # 因為 PPO update replay 會改 Net.hidden_m/w，更新完要恢復，才能接續下一段真實環境資料。
        rollout_end_hidden_m = clone_hidden(Net.hidden_m)
        rollout_end_hidden_w = clone_hidden(Net.hidden_w)
        rollout_end_goals = clone_tensor_list(goals)
        rollout_end_states = clone_tensor_list(states)
        rollout_end_masks = clone_tensor_list(masks)

        with torch.no_grad():
            _, _, _, next_v_m, next_v_w, _ = Net(
                state_tensor, obs_tensor, goals, states, masks[-1], save=False
            )
            bootstrap_values = {
                'next_v_m': next_v_m.detach(),
                'next_v_w': next_v_w.detach(),
            }

        # =========================================================
        # 2) PPO/MAPPO-style repeated update on the same rollout
        # =========================================================
        # 這裡才是真正「同一批資料多次更新」：
        # 每個 epoch 都從 rollout 初始 hidden/history 重新 forward，
        # 用 new_logp 與 old_logp 做 PPO clipped objective。
        for epoch in range(args.ppo_epoch):
            total_loss, loss_dict = FG_loss(Net, rollout, bootstrap_values, args)
            # total_loss, loss_dict = FG_loss_worker(Net, rollout, bootstrap_values, args)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(Net.parameters(), args.grad_clip)
            optimizer.step()

        logger.log_scalars(loss_dict, global_steps)

        # 恢復真實環境 rollout 結束後的 recurrent 狀態，避免被 update replay 污染。
        Net.hidden_m = detach_hidden(rollout_end_hidden_m)
        Net.hidden_w = detach_hidden(rollout_end_hidden_w)
        goals = clone_tensor_list(rollout_end_goals)
        states = clone_tensor_list(rollout_end_states)
        masks = clone_tensor_list(rollout_end_masks)

        if (episode // args.save_model_ep) > (last_saved_episode // args.save_model_ep):
            print(f"--- Saving model at episode: {episode} ---")
            os.makedirs(f'models/{args.env_name}', exist_ok=True)
            torch.save(Net.state_dict(), f'models/{args.env_name}/{args.env_name}_{args.run_name}_episode={episode}.pt')
            last_saved_episode = episode

    pbar.close()
    env.close()
    os.makedirs(f'models/{args.env_name}', exist_ok=True)
    torch.save(Net.state_dict(), f'models/{args.env_name}/{args.env_name}_{args.run_name}_episode={episode}.pt')


def main(args):
    run_name = args.run_name
    args.run_name = f"{run_name}"
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
    experiment(args)


if __name__ == '__main__':
    main(args)
