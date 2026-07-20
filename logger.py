import logging
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_name, args):
        dt = datetime.now()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_name = f"logs/train/{args.env_name}/{timestamp}_{run_name}"
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists('logs'):
            os.makedirs('logs')
            os.makedirs('models')

        self.writer = SummaryWriter(self.log_name)

        # logging.basicConfig(
        #     level=logging.DEBUG,
        #     format='%(asctime)s %(message)s',
        #     handlers=[
        #         logging.StreamHandler(),
        #         logging.FileHandler(f'{self.log_name}.log'),
        #         ],
        #     datefmt='%Y/%m/%d %I:%M:%S %p')

        # logging.info(args)


    def log_scalars(self, scalar_dict, step):
        for key, val in scalar_dict.items():
            self.writer.add_scalar(key, val, step)

    def log_episode(self, info, step):
        num_envs = info['train/episodic_reward'].shape[0]
        for i in range(num_envs):
            episodic_reward = info['train/episodic_reward'][i]
            if episodic_reward is not None:         # episode is done
                # print("episodic_reward:", episodic_reward)
                episode_dict = {
                    k: (info[k][i] if isinstance(info[k], np.ndarray) else info[k])
                    for k in info
                }
                
                self.n_eps += 1
                # print(episode_dict)
                self.log_scalars(episode_dict, step)
                # reward = episode_dict['returns/episodic_reward']
                # length = episode_dict['returns/episodic_length']
                # f_reward = float(episode_dict['returns/final_reward'])
                # print(reward)
                # time_expired = (time.time()-self.start_time) / 3600

                # logging.info(f"> ep = {self.n_eps} | total steps = {step}"
                #              f" | reward = {reward} | final reward = {f_reward}"
                #              f" | length = {length} | hours = {time_expired:.3f}")
            step += 1

    def log_eval_episode(self, info, step):
        num_envs = info['returns/episodic_reward'].shape[0]
        for i in range(num_envs):
            episodic_reward = info['returns/episodic_reward'][i]
            if episodic_reward is not None:         # episode is done
                # print("episodic_reward:", episodic_reward)
                episode_dict = {
                    k: (info[k][i] if isinstance(info[k], np.ndarray) else info[k])
                    for k in info
                }
                
                self.n_eps += 1
                # print(episode_dict)
                self.log_scalars(episode_dict, step)
                # reward = episode_dict['returns/episodic_reward']
                # length = episode_dict['returns/episodic_length']
                # f_reward = float(episode_dict['returns/final_reward'])
                # print(reward)
                # time_expired = (time.time()-self.start_time) / 3600

                # logging.info(f"> ep = {self.n_eps} | total steps = {step}"
                #              f" | reward = {reward} | final reward = {f_reward}"
                #              f" | length = {length} | hours = {time_expired:.3f}")
            step += 1

    def log_image(self, name, attn_tensor, step):
        """
        專門用來 log attention weights 熱力圖。
        attn_tensor: shape = (BA, D) 或 (batch, feature)
        """

        # Normalize 到 0~1 (避免 TensorBoard 顯示全黑)
        attn = attn_tensor[:9]  # (N, D)
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # 變成 (1, H, W) = (1, BA, D)
        attn_img = attn_norm.unsqueeze(0)

        # Log image
        self.writer.add_histogram(name, attn_img, step)


    def log_histogram(self, name, attn_tensor, step):
        """
        專門用來 log attention weights histogram。
        attn_tensor: shape = (BA, D)
        """
        attn = attn_tensor[:3]  # (N, D)
        self.writer.add_histogram(f"{name}0", attn[0], step)
        self.writer.add_histogram(f"{name}1", attn[1], step)
        self.writer.add_histogram(f"{name}2", attn[2], step)