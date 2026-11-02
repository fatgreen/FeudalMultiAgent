import numpy as np
import torch
from torch import nn
from torch.nn.functional import cosine_similarity as d_cos, normalize

from utils import init_hidden, weight_init
# from preprocess import Preprocessor
from dilated_lstm import DilatedLSTM

# from VIB_layer import *


class FMANetwork(nn.Module):
    def __init__(self,
                 num_envs,
                 num_agents,
                 input_dim_G,
                 input_dim_L,
                 hidden_dim_manager,
                 hidden_dim_worker,
                 n_actions,
                 time_horizon=10,
                 dilation=10,
                 device='cpu',
                 args=None):
        """
        naming convention inside the FeudalNetwork is selected
        to match paper variable naming convention.
        """

        super().__init__()
        self.envs = num_envs
        self.a = num_agents
        self.h = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.r = dilation
        self.n_actions = n_actions
        self.a_is_multi_discrete = args.a_is_multi_discrete
        self.action_dims = n_actions if isinstance(n_actions, (list, tuple)) else [n_actions]
        self.total_actions = sum(self.action_dims)
        self.g_detach = args.g_detach
        self.device = device
        
        self.percept_G = Perception(input_dim_G, self.d)
        self.percept_L = Perception(input_dim_L, self.d)
        self.percept_L4G = Perception(input_dim_L, self.d)
        self.percept_G4L = Perception(input_dim_G, self.d)
        self.manager = Manager(self.envs, self.a, self.h, self.d, self.r, args, device)
        self.worker = Worker(self.envs, self.a, self.h, self.d, self.k, self.action_dims, args.network, self.a_is_multi_discrete, device)
        self.hidden_m = init_hidden(self.envs*self.a, self.r * self.d,  # 10 * 256
                                    device=device, grad=False)
        self.hidden_w = init_hidden(self.envs*self.a, self.k * self.total_actions,
                                    device=device, grad=False)
        self.args = args
        self.to(device)
        self.apply(weight_init)
        # print(f'RNN mode: {self.mode}')

    def forward(self, x_G, x_L, goals, states, mask, save=True):
        """A forward pass through the whole feudal network.

        Order of operations:
        1. input goes through a preprocessor to normalize and put on device
        2. normalized input goes to the perception module resulting in a state
        3. state is input for manager which produces a goal
        4. state and goal is both input for worker which produces an action
           distribution.

        Args:
            x (np.ndarray): observation from the environment
            goals (list):  list of goal tensors, length = 2 * r + 1
            states (list): list of state tensors, length = 2 * r + 1
            mask (tensor): mask describing for each worker if episode is done.
            save (bool, optional): If we are calculating next_v, we do not
                                   store rnn states. Defaults to True.
        """
        # print("x_G shape:", x_G.shape)   # (envs, agents, (agents+1) * obs_dim)
        # print("x_L shape:", x_L.shape)   # (envs, agents, obs_dim)
        z_G = self.percept_G(x_G)
        z_L = self.percept_L(x_L)
        z_G4L = self.percept_G4L(x_G)
        if self.args.use_attention:
            z_L4G = self.percept_L4G(x_L)
        else:
            z_L4G = None
        goal, hidden_m, state_m, value_m, attn_w = self.manager(
            z_G, z_L4G, self.hidden_m, mask, save)

        # Ensure that we only have a list of size 2*h + 1, and we use FiLo
        if len(goals) == (2 * self.h + 1):
            goals.pop(0)
            states.pop(0)
        goals.append(goal)
        states.append(state_m.detach())  # states never have gradients active
        # The manager is ahead at least c steps, so we feed
        # only the first c+1 states to worker
        # action_dist, hidden_w, value_w = self.worker(  # worker doesn't need to update percept
        #     z_L, goals[:self.h + 1], self.hidden_w, mask, save)  # z.detach()
        action_dist, hidden_w, value_w = self.worker(  # worker doesn't need to update percept
            z_L, z_G4L, goal, self.hidden_w, mask, self.g_detach, save)  # z.detach()

        if save:
            # Optional, don't do this for the next_v
            self.hidden_m = hidden_m
            self.hidden_w = hidden_w

        return action_dist, goals, states, value_m, value_w, attn_w

    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)

    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)

    def repackage_hidden(self):
        def repackage_rnn(x):
            return [item.detach() for item in x]

        self.hidden_w = repackage_rnn(self.hidden_w)
        self.hidden_m = repackage_rnn(self.hidden_m)

    def init_obj(self):
        template = torch.zeros(self.envs*self.a, self.d)
        goals = [torch.zeros_like(template).to(self.device) for _ in range(2*self.h+1)]
        states = [torch.zeros_like(template).to(self.device) for _ in range(2*self.h+1)]
        masks = [torch.ones(self.envs*self.a, 1).to(self.device) for _ in range(2*self.h+1)]
        return goals, states, masks

# region perception
class Perception(nn.Module):  # feature extractor (CNN)
    def __init__(self, input_dim, d):
        super().__init__()
        self.percept = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d),
            nn.ReLU())

    def forward(self, x):
        # print("x shape:", x.shape)
        return self.percept(x)
        

# region Manager
class Manager(nn.Module):
    def __init__(self, envs, a, h, d, r, args, device):
        super().__init__()
        self.envs = envs
        self.a = a  # number of agents
        self.h = h  # Time Horizon
        self.d = d  # Hidden dimension size
        self.r = r  # Dilation level
        self.eps = args.eps
        self.device = device

        # self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn = DilatedLSTM(self.d, self.d, self.r)
        # self.Mrnn = nn.LSTMCell(self.d, self.d)
        self.att = nn.MultiheadAttention(self.d, num_heads=4, batch_first=True)
        """ SENet channel attention"""
        self.elem_att = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
            nn.Sigmoid()
            )
        self.fuse_ln = nn.LayerNorm(self.d)

        # self.ib1 = InformationBottleneck(self.d, kl_mult=args.kml, masking=False)
        self.critic = nn.Linear(self.d, 1)

        # self.count = 0
        self.prev_goal = None

    def forward(self, z, z_L4G, hidden, mask, save):
        # print("mask shape:", mask.shape)    # (envs*agents, 1)
        hidden = (mask * hidden[0], mask * hidden[1])
        # print("manager state shape:", state.shape)
        if z_L4G is not None:
            # if self.count == 0:
            #     print("Using Attention Mechanism in Manager")
            B, A, D = z_L4G.shape
            z = z.repeat_interleave(self.a, dim=0)      # (envs*agents, d)
            z_L4G  = z_L4G.reshape(B * A, D)      # env*agents, d
            
            # # ---------------SA
            # seq = torch.stack([z, z_L4G], dim=1)    # (envs*agents, 2, d)
            # attn_out, attn_w = self.att(seq, seq, seq, average_attn_weights=False)
            # attn_out = attn_out[:, 0, :]                      # (B*A, D)
            # attn_w_z_zL = attn_w[:, :, 0, 1]    # (B*A, heads)


            # GRN element-wise attention
            cat_input = torch.cat([z, z_L4G], dim=-1)   # (BA, 2D)
            alpha = self.elem_att(cat_input)            # (BA, D)
            # print("cat_input shape:", cat_input.shape)
            # print("alpha shape:", alpha.shape)

            #--element-wise attention combine
            attn_out = z + alpha * (z_L4G - z)    # (BA, D)
            attn_out = self.fuse_ln(attn_out)

            # no attention weights like MHA, but return alpha as attn weights
            attn_w_z_zL = alpha[:3].detach()        # (3, D)

            goal_hat, hidden = self.Mrnn(attn_out, hidden, save)
            # goal_ib = self.ib1(goal_hat)
            state_m = attn_out.detach()
            
        else:
            # if self.count == 0:
            #     print("Not using Attention Mechanism in Manager")
            z = z.reshape(self.envs*self.a, self.d)      # (envs*agents, d)
            # print("z shape:", z.shape)                  # (envs*agents, d)
            goal_hat, hidden = self.Mrnn(z, hidden, save)
            # goal_ib = self.ib1(goal_hat)
            state_m = z.detach()
            attn_w_z_zL = [[0.0]]

        value_est = self.critic(goal_hat)
        goal = normalize(goal_hat)

        # value_est = self.critic(goal_ib)
        # goal = normalize(goal_ib)

        # if self.prev_goal is None:
        #     self.prev_goal = torch.zeros_like(goal, requires_grad=False)

        # # Input LSTM every r steps
        # if self.count % self.r == 0:
        #     self.prev_goal = goal.detach()
        #     self.count = 0
        # else:
        #     goal = self.prev_goal
        
        
        # print("===goal shape:", goal.shape)        # (envs*agents, d)

        if (self.eps > torch.rand(1, device=self.device)[0]):
            # To encourage exploration in transition policy,
            # at every step with a small probability ε
            # we emit a random goal sampled from a uni-variate Gaussian.
            goal = torch.randn_like(goal, requires_grad=False)

        # use kl divergence for vib loss
        # self.vib_loss = self.ib1.kld

        # self.count += 1
        return goal, hidden, state_m, value_est, attn_w_z_zL

    def state_goal_cosine(self, states, goals, masks):
        """For the manager, we update using the cosine of:
            cos( S_{t+c} - S_{t}, G_{t} )

        Remember that states, goals are of size c * 2 + 1, with our current
        update time step right in the middle at t = c + 1.
        States should not have a gradient active, but goals[t] _should_.

        Args:
            states ([type]): list of size 2*C + 1, each element B x D
            goals ([type]): list of size 2*C + 1, each element B x D

        Returns:
            [type]: cosine distance between:
                        the difference state s_{t+c} - s_{t},
                        the goal embedding at timestep t g_t(theta).
        """
        # print(" ===> init states[t] shape:", states[0].shape)   # (e*a, 256)
        # print(" ===> init goals[t] shape:", goals[0].shape)   # (e*a, 256)
        # print(" ===> init masks:", masks[0].shape)   # (e*a, 1)
        t = self.h
        mask = torch.stack(masks[t: t + self.h - 1]).prod(dim=0)
        # print(" ===> mask shape:", mask.shape)   # (e*a, 1)
        cosine_dist = d_cos(states[t + self.h] - states[t], goals[t])
        # print(" ===> cosine_dist shape:", cosine_dist.shape)   # (e*a)
        cosine_dist = mask * cosine_dist.unsqueeze(-1)
        return cosine_dist


# region Worker
class Worker(nn.Module):
    def __init__(self, envs, a, h, d, k, num_actions, net_mode, a_is_multi_discrete, device):
        super().__init__()
        self.envs = envs
        self.a = a
        self.h = h
        self.d = d
        self.k = k
        self.net_mode = net_mode
        self.a_is_multi_discrete = a_is_multi_discrete
        self.device = device

        # num_actions: [5] or [5, 10]
        self.action_dims = num_actions if isinstance(num_actions, (list, tuple)) else [num_actions]
        self.total_actions = sum(self.action_dims)
        
        self.Wrnn = nn.LSTMCell(d, k * self.total_actions)
        # self.goal_encoder = nn.Linear(d, d)
        self.goal_obs_encoder = nn.Linear(2*d, d) if not net_mode == "BASE" else None
        self.phi = nn.Linear(d, k, bias=False)

        if net_mode == "Ghead":
            if not self.a_is_multi_discrete:
                self.action_head = nn.Linear(self.k * self.total_actions, self.action_dims[0])
            else:
                self.action_heads = nn.ModuleList([
                    nn.Linear(self.k * self.total_actions, dim) for dim in self.action_dims
                ])
        else:
            self.action_head = None
            self.action_heads = None

        self.elem_att = nn.Sequential(
            nn.Linear(self.d * 2, self.d),
            nn.ReLU(),
            nn.Linear(self.d, self.d),
            nn.Sigmoid()
            )
        self.fuse_ln = nn.LayerNorm(self.d)

        self.critic = nn.Sequential(
            nn.Linear(k * self.total_actions + d, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z, z_G4L, goal, hidden, mask, g_detach, save):
        """[summary]

        Args:
            z ([type]): latent representation from perception, size B x D
            z_G4L ([type]): latent representation from goal perception, size B x D
            goals ([type]): list of length C, with each tensor of size B x D
            hidden ([type]): recurrent cells, size B x (K*num_actions)

        Returns:
            [type]: returns the policy: a normalized distribution over
            actions, hidden cells (u, cx) and the value estimation
            specific to the worker.
        """
        mask = mask.reshape(-1, 1)      # (envs * agents, 1)
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_d = goal.detach() if g_detach else goal   # (envs*agents, d)

        z = z.reshape(-1, z.size(-1))
        if not self.net_mode == "BASE":                  # GOG, Ghead
            # z_g = self.goal_obs_encoder(torch.cat([z, goal_d], dim=-1))

            cat_input = torch.cat([z, goal_d], dim=-1)   # (BA, 2D)
            alpha = self.elem_att(cat_input)            # (BA, D)
            # print("cat_input shape:", cat_input.shape)
            # print("alpha shape:", alpha.shape)

            #--element-wise attention combine
            attn_out = z + alpha * (goal_d - z)    # (BA, D)
            z_g = self.fuse_ln(attn_out)
        else:
            z_g = z

        (u, cx) = self.Wrnn(z_g, hidden)
        hidden = (u, cx)
        
        # value_est = self.critic(u)
        z_G4L = z_G4L.repeat_interleave(self.a, dim=0)
        # value_est = self.critic(torch.cat([u, z_G4L, goal_d.detach()], dim=-1))
        value_est = self.critic(torch.cat([u, z_G4L], dim=-1))

        # BASE / GOG: use phi(goal) and einsum
        if self.net_mode != "Ghead":
            w = self.phi(goal_d)  # (B, k)
            u_3d = u.reshape(u.shape[0], self.k, self.total_actions)  # (B, k, total_actions)
            logits_all = torch.einsum("bk,bka->ba", w, u_3d)  # (B, total_actions)

            if not self.a_is_multi_discrete:
                probs = logits_all.softmax(dim=-1)
                probs = probs.reshape(self.envs, self.a, self.action_dims[0])
                return probs, hidden, value_est
            else:
                split_logits = torch.split(logits_all, self.action_dims, dim=-1)
                probs_list = [
                    logits.softmax(dim=-1).reshape(self.envs, self.a, dim)
                    for logits, dim in zip(split_logits, self.action_dims)
                ]
                return probs_list, hidden, value_est

        # Ghead: explicit action heads
        else:
            if not self.a_is_multi_discrete:
                logits = self.action_head(u)
                probs = logits.softmax(dim=-1)
                probs = probs.reshape(self.envs, self.a, self.action_dims[0])
                return probs, hidden, value_est
            else:
                probs_list = []
                for head, dim in zip(self.action_heads, self.action_dims):
                    logits = head(u)
                    probs = logits.softmax(dim=-1)
                    probs = probs.reshape(self.envs, self.a, dim)
                    probs_list.append(probs)
                return probs_list, hidden, value_est
    # region Intrinsic Reward
    def intrinsic_reward(self, states, goals, masks):
        """To calculate the intrinsic reward for the Worker (Eq. 8),
        we look at the horizon C, and for each horizon step i, we
        take current state s_t minus horizon state s_{t-i} and
        calculate how similar it is to the goal at timestep s_{t-i}.

        Args:
            states: states from the Manager, a list of size C,
                    with each tensor of size B x D
            goals: goal vectors from the Manager, a list of size C,
                    with each tensor of size  B x D

        remember: our lists are of length c*2 + 1, with c + 1 being the exact
        middle, and we use this as our reference point for time step t.

        We hence calculate going backwards through the list of states, goals,
        state[t] == state[c+1]
        state[t] - state[t - 1], goal[t - 1]
        state[t] - state[t - 2], goal[t - 2]
        ....
        state[t] - state[t - c], goal[t - c]

        Returns:
            Intrinsic reward for the Worker
        """
        t = self.h
        r_i = torch.zeros(self.envs*self.a, 1).to(self.device)
        mask = torch.ones(self.envs*self.a, 1).to(self.device)
        # print(" ===> init mask shape:", mask.shape)   # (e*a, 1)
        # print(" ===> init states[t] shape:", states[t].shape)   # (e*a, 256)
        # print(" ===> init goals[t] shape:", goals[t].shape)   # (e*a, 256)
        # print(" ===> init masks:", masks[t].shape)   # (e*a, 1)
        for i in range(1, self.h + 1):
            # print("i :", i)
            # print(" ===> states[t] shape:", states[t].shape)   # (e*a, 256)
            # print(" ===> goals[t - i] shape:", goals[t - i].shape)   # (e*a, 256)
            r_i_t = d_cos(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i_t = 0.5 * (r_i_t + 1.0)  # scale to [0, 1]
            # print(" ===> r_i_t shape:", r_i_t.shape)   # (e*a, 1)
            # print(" ===> mask shape:", mask.shape)   # (e*a, 1)
            r_i += (mask * r_i_t)
            mask = mask * masks[t - i]

        r_i = r_i.detach()
        # print(" ===> intrinsic reward shape:", r_i.shape)   # (envs, agents)
        return r_i / self.h


# region Full Loss function
def _clone_tensor_list(xs):
    """Clone goal/state/mask history for recurrent PPO replay."""
    return [x.detach().clone() for x in xs]


def _clone_hidden(hidden):
    """Clone Net.hidden_m / Net.hidden_w. hidden is usually [h, c] or (h, c)."""
    return [h.detach().clone() for h in hidden]


def _policy_logp_entropy(action_dist, action, args, eps=1e-8):
    """
    Re-compute log_prob and entropy of the rollout action under current policy.

    PPO/MAPPO does NOT sample new actions during update.
    It evaluates the old rollout action with the current policy and compares
    new_logp against old_logp.
    """
    if args.a_is_multi_discrete:
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, dtype=torch.long, device=args.device)
        action = action.long().to(args.device)

        logp = 0.0
        entropy = 0.0
        for branch_id, probs in enumerate(action_dist):
            probs = probs.clamp(min=eps, max=1.0)
            branch_action = action[:, :, branch_id].unsqueeze(-1)
            branch_logp = torch.log(probs.gather(-1, branch_action))
            branch_entropy = -(probs * torch.log(probs)).sum(dim=-1, keepdim=True)
            logp = logp + branch_logp
            entropy = entropy + branch_entropy

        return logp.reshape(args.num_envs * args.num_agents, 1), entropy.reshape(args.num_envs * args.num_agents, 1)

    else:
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


def _stack_rollout(rollout, key):
    return torch.stack(rollout[key], dim=0)


def _normalize_tensor(x, eps=1e-8):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def _compute_targets_and_advantages(rollout, bootstrap_values, args):
    """
    Compute returns and advantages once from rollout data.

    Difference from the old version:
    - old version used storage.v_m/storage.v_w directly for repeated backward.
    - this version only uses old values to compute fixed advantages/targets.
      During every PPO epoch, values are re-computed by the current Net.
    """
    next_v_m = bootstrap_values['next_v_m'].detach()
    next_v_w = bootstrap_values['next_v_w'].detach()

    ret_m = next_v_m
    ret_w = next_v_w
    ret_m_list = [None for _ in range(args.num_steps)]
    ret_w_list = [None for _ in range(args.num_steps)]

    for t in reversed(range(args.num_steps)):
        r_t = rollout['r'][t]
        m_t = rollout['m'][t]
        ret_m = r_t + args.gamma_m * ret_m * m_t
        ret_w = r_t + args.gamma_w * ret_w * m_t
        ret_m_list[t] = ret_m.detach()
        ret_w_list[t] = ret_w.detach()

    ret_m_storage = torch.stack(ret_m_list, dim=0)
    ret_w_storage = torch.stack(ret_w_list, dim=0)

    # Keep the same behavior as your original FG_loss: normalize returns.
    ret_m_storage = _normalize_tensor(ret_m_storage)
    ret_w_storage = _normalize_tensor(ret_w_storage)

    rewards_intrinsic = _stack_rollout(rollout, 'r_i').detach()
    old_v_m = _stack_rollout(rollout, 'old_v_m').detach()
    old_v_w = _stack_rollout(rollout, 'old_v_w').detach()

    target_m = ret_m_storage
    target_w = ret_w_storage + args.alpha * rewards_intrinsic

    advantage_m = (target_m - old_v_m).detach()
    advantage_w = (target_w - old_v_w).detach()

    # PPO/MAPPO normally normalizes advantages. Default: enabled.
    if getattr(args, 'use_adv_norm', True):
        advantage_m = _normalize_tensor(advantage_m)
        advantage_w = _normalize_tensor(advantage_w)

    # Optional safety clipping, only used if args.adv_clip exists.
    if hasattr(args, 'adv_clip') and args.adv_clip is not None:
        advantage_m = torch.clamp(advantage_m, -args.adv_clip, args.adv_clip)
        advantage_w = torch.clamp(advantage_w, -args.adv_clip, args.adv_clip)

    return {
        'target_m': target_m.detach(),
        'target_w': target_w.detach(),
        'advantage_m': advantage_m.detach(),
        'advantage_w': advantage_w.detach(),
        'rewards_intrinsic': rewards_intrinsic.detach(),
    }


def _recompute_sequence(Net, rollout, args):
    """
    Re-play the rollout inputs through the current network.

    This is the key MAPPO-style change:
    - old_logp/action/rewards are fixed from rollout
    - new_logp/value/cosine are produced by the current policy every epoch
    """
    Net.hidden_m = _clone_hidden(rollout['init_hidden_m'])
    Net.hidden_w = _clone_hidden(rollout['init_hidden_w'])
    goals = _clone_tensor_list(rollout['init_goals'])
    states = _clone_tensor_list(rollout['init_states'])
    masks = _clone_tensor_list(rollout['init_masks'])

    new_logps = []
    entropies = []
    value_ms = []
    value_ws = []
    state_goal_cosines = []

    for t in range(args.num_steps):
        state_t = rollout['state'][t].to(args.device)
        obs_t = rollout['obs'][t].to(args.device)
        action_t = rollout['action'][t].to(args.device)

        action_dist, goals, states, value_m, value_w, _ = Net(
            state_t, obs_t, goals, states, masks[-1], save=True
        )
        new_logp, entropy = _policy_logp_entropy(action_dist, action_t, args)

        # Match rollout collection order:
        # forward -> env.step -> append new mask -> compute intrinsic/cosine.
        masks.pop(0)
        masks.append(rollout['m'][t].to(args.device).detach())

        new_logps.append(new_logp)
        entropies.append(entropy)
        value_ms.append(value_m)
        value_ws.append(value_w)
        state_goal_cosines.append(Net.state_goal_cosine(states, goals, masks))

    return {
        'new_logp': torch.stack(new_logps, dim=0),
        'entropy': torch.stack(entropies, dim=0),
        'value_m': torch.stack(value_ms, dim=0),
        'value_w': torch.stack(value_ws, dim=0),
        'state_goal_cos': torch.stack(state_goal_cosines, dim=0),
    }


def FG_loss(Net, rollout, bootstrap_values, args):
    """
    MAPPO/PPO-style full-batch repeated update for your Feudal network.

    Main changes from the old FG_loss:
    1. The function no longer receives Storage with old graph tensors.
    2. old_logp/old_value are detached rollout data.
    3. Every PPO epoch re-computes new_logp/new_value/new_state_goal_cos
       from the current Net.
    4. Worker actor loss uses PPO clipped surrogate objective.
    5. Manager actor loss is still your Feudal cosine objective, but it is
       computed from the current Net each epoch instead of stale rollout graph.
    """
    computed = _compute_targets_and_advantages(rollout, bootstrap_values, args)
    replay = _recompute_sequence(Net, rollout, args)

    old_logp = _stack_rollout(rollout, 'old_logp').detach().to(args.device)
    new_logp = replay['new_logp']
    entropy = replay['entropy'].mean()

    target_m = computed['target_m'].to(args.device)
    target_w = computed['target_w'].to(args.device)
    advantage_m = computed['advantage_m'].to(args.device)
    advantage_w = computed['advantage_w'].to(args.device)
    rewards_intrinsic = computed['rewards_intrinsic'].to(args.device)

    value_m = replay['value_m']
    value_w = replay['value_w']
    state_goal_cosines = replay['state_goal_cos']

    # -----------------------------
    # Worker PPO clipped actor loss (clipped surrogate objective)
    # -----------------------------
    clip_param = getattr(args, 'ppo_clip', getattr(args, 'clip_param', 0.2))
    ratio = torch.exp(new_logp - old_logp)
    surr1 = ratio * advantage_w
    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage_w
    loss_worker = -torch.min(surr1, surr2).mean()

    # -----------------------------
    # Manager Feudal actor loss
    # -----------------------------
    # Manager is not a categorical action policy here, so we keep your cosine
    # policy-gradient-like objective, but re-compute cosine every epoch.
    loss_manager = (-state_goal_cosines * advantage_m).mean()

    # -----------------------------
    # Critic losses use current value estimates
    # -----------------------------
    value_m_loss = 0.5 * (target_m - value_m).pow(2).mean()
    value_w_loss = 0.5 * (target_w - value_w).pow(2).mean()

    manager_loss = loss_manager + value_m_loss
    worker_loss = loss_worker + value_w_loss - args.entropy_coef * entropy
    total_loss = worker_loss + manager_loss

    approx_kl = (old_logp - new_logp).mean().detach()
    clip_fraction = ((ratio - 1.0).abs() > clip_param).float().mean().detach()

    return total_loss, {
        'loss/total_fun_loss': total_loss.item(),
        'loss/worker': loss_worker.item(),
        'loss/manager': loss_manager.item(),
        'loss/value_worker': value_w_loss.item(),
        'loss/value_manager': value_m_loss.item(),
        'loss/total_worker_loss': worker_loss.item(),
        'loss/total_manager_loss': manager_loss.item(),
        'worker/entropy': entropy.item(),
        'worker/advantage': advantage_w.mean().item(),
        'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
        'worker/ratio': ratio.mean().item(),
        'worker/approx_kl': approx_kl.item(),
        'worker/clip_fraction': clip_fraction.item(),
        'manager/cosines': state_goal_cosines.mean().item(),
        'manager/advantage': advantage_m.mean().item(),
        'average_episode_rewards': torch.stack(rollout['r'], dim=0).mean().item() * args.num_steps,
    }


# region Worker-only Loss function
# def _recompute_worker_only(Net, rollout, args):
#     """
#     Fast PPO replay:
#     - 不重跑 Manager / DilatedLSTM
#     - 使用 rollout 時固定下來的 goal.detach()
#     - 只重算 Worker 的 new_logp / entropy / value_w
#     """
#     new_logps = []
#     entropies = []
#     value_ws = []

#     for t in range(args.num_steps):
#         z_L_t = rollout['z_L'][t].to(args.device)
#         goal_t = rollout['goal'][t].to(args.device)
#         action_t = rollout['action'][t].to(args.device)
#         mask_prev_t = rollout['mask_prev'][t].to(args.device)

#         hidden_w_t = [
#             h.to(args.device).detach()
#             for h in rollout['hidden_w'][t]
#         ]

#         action_dist, _, value_w = Net.worker(
#             z_L_t,
#             goal_t,
#             hidden_w_t,
#             mask_prev_t,
#             g_detach=True,
#             save=False
#         )

#         new_logp, entropy = _policy_logp_entropy(
#             action_dist,
#             action_t,
#             args
#         )

#         new_logps.append(new_logp)
#         entropies.append(entropy)
#         value_ws.append(value_w)

#     return {
#         'new_logp': torch.stack(new_logps, dim=0),
#         'entropy': torch.stack(entropies, dim=0),
#         'value_w': torch.stack(value_ws, dim=0),
#     }


# def FG_loss_worker(Net, rollout, bootstrap_values, args):
#     """
#     Worker-only PPO update for epoch > 0.

#     用途：
#     - epoch 0：用原本 FG_loss()，Manager + Worker 都更新
#     - epoch 1~N：用 FG_loss_worker()，只更新 Worker，速度快很多

#     注意：
#     - 這裡不更新 Manager
#     - goal 使用 rollout 時存下來的 goal.detach()
#     - old_logp 固定，new_logp 由目前 Worker 重新計算
#     """
#     computed = _compute_targets_and_advantages(
#         rollout,
#         bootstrap_values,
#         args
#     )

#     replay = _recompute_worker_only(
#         Net,
#         rollout,
#         args
#     )

#     old_logp = _stack_rollout(
#         rollout,
#         'old_logp'
#     ).detach().to(args.device)

#     new_logp = replay['new_logp']
#     entropy = replay['entropy'].mean()

#     target_w = computed['target_w'].to(args.device)
#     advantage_w = computed['advantage_w'].to(args.device)
#     rewards_intrinsic = computed['rewards_intrinsic'].to(args.device)

#     value_w = replay['value_w']

#     clip_param = getattr(
#         args,
#         'ppo_clip',
#         getattr(args, 'clip_param', 0.2)
#     )

#     # PPO clipped surrogate loss
#     ratio = torch.exp(new_logp - old_logp)

#     surr1 = ratio * advantage_w
#     surr2 = torch.clamp(
#         ratio,
#         1.0 - clip_param,
#         1.0 + clip_param
#     ) * advantage_w

#     loss_worker = -torch.min(surr1, surr2).mean()

#     # Worker critic loss
#     # 如果你要穩一點，可以改成 smooth_l1_loss
#     value_w_loss = 0.5 * (target_w - value_w).pow(2).mean()

#     worker_loss = (
#         loss_worker
#         + value_w_loss
#         - args.entropy_coef * entropy
#     )

#     approx_kl = (old_logp - new_logp).mean().detach()
#     clip_fraction = (
#         (ratio - 1.0).abs() > clip_param
#     ).float().mean().detach()

#     return worker_loss, {
#         'loss/total_fun_loss': worker_loss.item(),
#         'loss/worker': loss_worker.item(),
#         'loss/value_worker': value_w_loss.item(),
#         'loss/total_worker_loss': worker_loss.item(),
#         'worker/entropy': entropy.item(),
#         'worker/advantage': advantage_w.mean().item(),
#         'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
#         'worker/ratio': ratio.mean().item(),
#         'worker/approx_kl': approx_kl.item(),
#         'worker/clip_fraction': clip_fraction.item(),
#         'average_episode_rewards': torch.stack(
#             rollout['r'],
#             dim=0
#         ).mean().item() * args.num_steps,
#     }
