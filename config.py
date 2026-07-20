import argparse

game_list = ["simple_spread_v3"             # 0
        ,"simple_reference_v3"              # 1
        ,"simple_push_v3"                   # 2
        ,"simple_crypto_v3"                 # 3
        ,"simple_adversary_v3"              # 4
        ,"simple_speaker_listener_v4"       # 5
        ,"simple_tag_v3"                    # 6
        ,"simple_world_comm_v3"             # 7
]

net_list = [
    "BASE",     # 0
    "GOG",      # 1
    "Ghead"     # 2
]

def get_config():
    
    parser = argparse.ArgumentParser(description='FGNet', formatter_class=argparse.RawDescriptionHelpFormatter)
    # ENVIRONMENT PARAMETERS
#     parser.add_argument('--network', type=str, default=net_list[1],
#                         help='network architecture')
    parser.add_argument('--lr', type=float, default=7e-4,           # 7e-4
                        help='learning rate')
    parser.add_argument('--env_name', type=str, default=game_list[1],
                        help='gym environment name')
    parser.add_argument('--num_envs', type=int, default=64,                             #====================num envs 64====================
                        help='number of parallel environments to run')
    parser.add_argument('--num_eval_envs', type=int, default=1,                         #====================num envs 64====================
                        help='number of parallel environments to run evaluation')
    parser.add_argument('--num_agents', type=int, default=3,                            #====================num agents====================
                        help='number of agents')
    parser.add_argument('--max_steps', type=int, default=int(25),                       #====================max cycles====================
                        help='maximum number of training steps in total')
    parser.add_argument('--max_episodes', type=int, default=int(4e5),                   #====================4e5 = 10Msteps====================
                        help='maximum number of training episodes in total')
    parser.add_argument('--max_total_steps', type=int, default=int(3e6),                #====================3e6SR 4e7SS====================
                        help='maximum number of training steps in total')
    parser.add_argument('--test_episodes', type=int, default=int(8),
                        help='maximum number of testing episodes in total')

    # GENERIC RL/MODEL PARAMETERS
    parser.add_argument('--num_steps', type=int, default=25,                            #====================update steps====================
                        help='number of steps the agent takes before updating')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='Gradient clipping (recommended).')
    parser.add_argument('--adv_clip', type=float, default=5.,
                        help='Advantage clipping (recommended).')
    parser.add_argument('--entropy_coef', type=float, default=0.01,                     #====================0.01====================
                        help='Entropy coefficient to encourage exploration.')
    parser.add_argument('--ppo_epoch', type=int, default=1,
                        help='PPO epochs per update.')
    parser.add_argument('--ppo_clip', type=float, default=0.5,
                        help='PPO clipping parameter.')

    # NETWORK MODEL RELATED PARAMETERS
    parser.add_argument('--network', type=str, default=net_list[2],                     #====================network mode====================
                        help='network architecture')
    parser.add_argument('--g_detach', type=bool, default=True,
                        help='Detach goal from computation graph')
    parser.add_argument('--policy_sharing', type=bool, default=True,
                        help='Policy sharing among agents')
    parser.add_argument('--use_attention', type=bool, default=True,
                        help='Use attention mechanism')

    # SPECIFIC PARAMETERS
    parser.add_argument('--time_horizon', type=int, default=10,                         #====================time horizon 5====================
                        help='Manager horizon (c)')
    parser.add_argument('--hidden_dim_manager', type=int, default=256,
                        help='Hidden dim (d)')
    parser.add_argument('--hidden_dim_worker', type=int, default=16,
                        help='Hidden dim for worker (k)')
    parser.add_argument('--gamma_w', type=float, default=0.99,
                        help="discount factor worker")
    parser.add_argument('--gamma_m', type=float, default=0.999,
                        help="discount factor manager")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Intrinsic reward coefficient in [0, 1]')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='Random Gaussian goal for exploration')
    parser.add_argument('--dilation', type=int, default=10,                             #====================dilation 5====================
                        help='Dilation parameter for manager LSTM.')

    parser.add_argument('--ib_lr', type=float, default=1e-4,
                        help='Separate learning rate for information bottleneck params. Set to -1 to follow opt.lr.')
    parser.add_argument('--kl_fac', type=float, default=1e-6, help='Factor for the KL term.')
    parser.add_argument('--kml', type=float, default=2, help='Compression factor multiplier')

    # EXPERIMENT RELATED PARAMS
    parser.add_argument('--run_name', type=str, default=f'ATTN_{net_list[2]}_comparison5',   #====================ATTN_{net_list[0]}_MultiUpdate===================
                        help='run name for the logger.')
    parser.add_argument('--load_model', type=str, default='simple_reference_v3_ATTN_Ghead_comparison1_episode=120001',
                        help='model name to load.')
    parser.add_argument('--seed', type=int, default=123456,                                  #==================== 123456 ====================
                        help='reproducibility seed.')
    parser.add_argument('--evaluate', type=bool, default=False,
                        help='evaluate or not.')
    parser.add_argument('--render', type=str, default=None,
                        help='render mode.')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluation interval.')
    parser.add_argument('--window_size', type=int, default=100,
                        help='window size for moving average.')
    parser.add_argument('--save_model_ep', type=int, default=int(1e6),
                        help='number of episodes between saving the model.')
    
    # ENVRINMENTAL PARAMETERS
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3,
                        help="number of landmarks")
    parser.add_argument("--episode_length", type=int,
                        default=25, help="Max length for any episode")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--a_is_multi_discrete", type=bool,
                        default=False, help="Whether the action space is multi-discrete")
    

    args = parser.parse_args()

    return args