# FeudalMultiAgent

**Hierarchical Manager–Worker Multi-Agent Reinforcement Learning with PPO-Style Recurrent Training**

[English](#english) | [中文](#中文)

> [!NOTE]
> This repository is a research implementation of hierarchical cooperative multi-agent reinforcement learning.
> The root-level Feudal implementation is built on top of components adapted from the MAPPO/on-policy codebase.

---

# English

## Overview

**FeudalMultiAgent** implements a hierarchical Manager–Worker architecture for cooperative multi-agent reinforcement learning.

The high-level **Manager** integrates global team information and agent-specific local information to generate a latent goal for each agent. The low-level **Worker** combines its local observation with the goal and produces an action distribution.

The current implementation includes:

* Hierarchical Manager–Worker decision making
* Global and local observation encoders
* Element-wise gated feature fusion
* Dilated LSTM for long-horizon Manager reasoning
* Recurrent Worker policy
* Manager and Worker value functions
* Goal-based intrinsic rewards
* PPO/MAPPO-style repeated updates
* Discrete and MultiDiscrete action spaces
* Parameter sharing among agents
* TensorBoard logging, checkpoint saving, and rollout rendering

## Execution Assumption

> [!IMPORTANT]
> The current Manager receives concatenated observations/global information during action inference.
> Therefore, the current implementation uses **centralized information during execution** and is not a strict decentralized-execution CTDE implementation.

The architecture can be summarized as:

```text
Global information ──> Global encoder ──────────────┐
                                                     ├─> Manager ─> Goal
Agent local obs.  ──> Local encoder ─> gated fusion ┘                │
                                                                      v
Agent local obs.  ──> Local encoder ─> gated goal fusion ─> Worker LSTM
                                                                      │
                                                                      v
                                                               Action policy
```

The Worker critic additionally receives a global feature, providing centralized value estimation.

## Architecture

### 1. Perception Encoders

Global and local inputs are independently projected into latent representations using MLP encoders:

```text
Global input  -> percept_G   -> global latent feature
Local input   -> percept_L   -> local latent feature
Local input   -> percept_L4G -> local feature for the Manager
Global input  -> percept_G4L -> global feature for the Worker critic
```

### 2. Manager

For each agent, the Manager fuses global and local latent features with an element-wise gate:

```text
alpha_i = sigmoid(MLP([z_global, z_local_i]))

z_manager_i = LayerNorm(
    z_global + alpha_i * (z_local_i - z_global)
)
```

The fused feature is processed by a Dilated LSTM:

```text
goal_hat_i,t = DilatedLSTM(z_manager_i,t)

goal_i,t = normalize(goal_hat_i,t)
```

The Manager also estimates a high-level value:

```text
V_manager_i,t = Critic_manager(goal_hat_i,t)
```

The Manager objective encourages the generated goal to align with the future latent-state transition:

```text
cos(
    state_i,t+c - state_i,t,
    goal_i,t
)
```

### 3. Worker

The Worker combines its local representation and the Manager goal through another element-wise gate:

```text
beta_i = sigmoid(MLP([z_local_i, goal_i]))

z_worker_i = LayerNorm(
    z_local_i + beta_i * (goal_i - z_local_i)
)
```

The fused representation is processed by a recurrent Worker network:

```text
u_i,t = LSTM_worker(z_worker_i,t)
```

The Worker supports three network modes:

| Mode    | Description                                                                                        |
| ------- | -------------------------------------------------------------------------------------------------- |
| `BASE`  | Uses the local feature as the Worker recurrent input and applies goal-conditioned action scoring.  |
| `GOG`   | Uses gated observation–goal fusion and goal-conditioned action scoring.                            |
| `Ghead` | Uses gated observation–goal fusion followed by explicit action heads. This is the current default. |

The Worker critic uses both the recurrent Worker feature and a global feature:

```text
V_worker_i,t = Critic_worker([u_i,t, z_global])
```

### 4. Intrinsic Reward

The Worker receives an intrinsic reward based on the cosine similarity between recent latent-state transitions and previous Manager goals:

```text
r_intrinsic_i,t =
    mean_j cos(
        state_i,t - state_i,t-j,
        goal_i,t-j
    )
```

The cosine score is scaled to `[0, 1]` before averaging over the Manager horizon.

## Training Objective

Training is divided into rollout collection and repeated recurrent replay updates.

During rollout collection, actions, rewards, masks, old log probabilities, value estimates, recurrent states, goals, and latent states are stored without retaining the original computation graph.

During every PPO epoch, the rollout sequence is replayed through the current network to recompute:

* New action log probabilities
* Policy entropy
* Manager values
* Worker values
* Manager state–goal cosine scores

### Worker Objective

The Worker uses the PPO clipped surrogate objective:

```text
ratio_t = exp(new_log_prob_t - old_log_prob_t)

L_worker_actor =
    -mean(
        min(
            ratio_t * advantage_worker_t,
            clip(ratio_t, 1-epsilon, 1+epsilon)
            * advantage_worker_t
        )
    )
```

### Manager Objective

The Manager uses a Feudal cosine objective weighted by the Manager advantage:

```text
L_manager_actor =
    -mean(
        cosine_t * advantage_manager_t
    )
```

### Value Objectives

```text
L_manager_value =
    0.5 * mean(
        target_manager - V_manager
    )^2

L_worker_value =
    0.5 * mean(
        target_worker - V_worker
    )^2
```

The Worker target includes the intrinsic reward:

```text
target_worker =
    extrinsic_return_worker
    + intrinsic_reward_weight * r_intrinsic
```

The full loss is:

```text
L_total =
    L_manager_actor
    + L_manager_value
    + L_worker_actor
    + L_worker_value
    - entropy_coefficient * entropy
```

## Supported Environments

The root-level Feudal training and evaluation scripts are currently connected to the cooperative MPE environments below:

| Environment           | Agents | Action space                                                           |
| --------------------- | -----: | ---------------------------------------------------------------------- |
| `simple_spread_v3`    |      3 | Discrete movement actions, dimension `5`                               |
| `simple_reference_v3` |      2 | MultiDiscrete movement and communication actions, dimensions `[5, 10]` |

Although additional environments and baseline algorithms are present in the bundled `onpolicy/` directory, the root-level `train_mappo.py` and `evaluate.py` scripts currently recognize only these two environments.

## Example Rollouts

### Simple Spread

![Simple Spread rollout](records/simple_spread_v3/simple_spread_v3_ATTN_Ghead_comparison5_episode=1600001_seed123456.gif)

### Simple Reference

![Simple Reference rollout](records/simple_reference_v3/simple_reference_v3_ATTN_Ghead_comparison1_episode=120001_seed123456.gif)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fatgreen/FeudalMultiAgent.git
cd FeudalMultiAgent
```

### 2. Create a Python environment

Conda example:

```bash
conda create -n feudal-marl python=3.10
conda activate feudal-marl
```

Virtual environment example:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install PyTorch

Install a PyTorch build compatible with your operating system and CUDA version before installing the remaining dependencies.

Example for a CPU installation:

```bash
pip install torch torchvision
```

For CUDA builds, use the installation command recommended by the official PyTorch installation selector.

### 4. Install dependencies

```bash
pip install -r requirement.txt
pip install -e .
```

> [!NOTE]
> `requirement.txt` records the development environment and contains tightly pinned versions.
> Some packages may require platform-specific adjustments.

## Training

### Simple Reference

```bash
python train_mappo.py \
    --env_name simple_reference_v3 \
    --run_name reference_ghead
```

Windows PowerShell or Command Prompt:

```powershell
python train_mappo.py --env_name simple_reference_v3 --run_name reference_ghead
```

### Simple Spread

```bash
python train_mappo.py \
    --env_name simple_spread_v3 \
    --run_name spread_ghead
```

Windows PowerShell or Command Prompt:

```powershell
python train_mappo.py --env_name simple_spread_v3 --run_name spread_ghead
```

### Example with Additional Parameters

```bash
python train_mappo.py \
    --env_name simple_reference_v3 \
    --run_name reference_h10 \
    --num_envs 64 \
    --max_total_steps 3000000 \
    --num_steps 25 \
    --ppo_epoch 3 \
    --ppo_clip 0.2 \
    --time_horizon 10 \
    --dilation 10 \
    --lr 0.0007
```

## Evaluation

Models are loaded from:

```text
models/<environment>/<model_name>.pt
```

Pass the model name without the `.pt` extension:

```bash
python evaluate.py \
    --env_name simple_reference_v3 \
    --load_model simple_reference_v3_reference_ghead_episode=120001 \
    --test_episodes 8
```

Windows example:

```powershell
python evaluate.py --env_name simple_reference_v3 --load_model simple_reference_v3_reference_ghead_episode=120001 --test_episodes 8
```

Rendered rollouts are saved under:

```text
records/<environment>/
```

## TensorBoard

Training logs are written under:

```text
logs/train/<environment>/
```

Start TensorBoard with:

```bash
tensorboard --logdir logs/train
```

Then open the local TensorBoard address shown in the terminal.

## Important Configuration Options

Most experiment settings are defined in `config.py`.

| Argument               |               Default | Description                                        |
| ---------------------- | --------------------: | -------------------------------------------------- |
| `--env_name`           | `simple_reference_v3` | Training environment                               |
| `--network`            |               `Ghead` | Worker architecture: `BASE`, `GOG`, or `Ghead`     |
| `--num_envs`           |                  `64` | Number of parallel environments                    |
| `--max_total_steps`    |             `3000000` | Total environment steps                            |
| `--num_steps`          |                  `25` | Rollout length before an update                    |
| `--ppo_epoch`          |                   `1` | Number of repeated updates per rollout             |
| `--ppo_clip`           |                 `0.5` | PPO clipping range                                 |
| `--lr`                 |                `7e-4` | Learning rate                                      |
| `--time_horizon`       |                  `10` | Manager horizon                                    |
| `--dilation`           |                  `10` | Dilated LSTM cycle                                 |
| `--hidden_dim_manager` |                 `256` | Manager latent dimension                           |
| `--hidden_dim_worker`  |                  `16` | Worker action-embedding dimension                  |
| `--gamma_m`            |               `0.999` | Manager discount factor                            |
| `--gamma_w`            |                `0.99` | Worker discount factor                             |
| `--alpha`              |                 `0.5` | Intrinsic reward coefficient                       |
| `--entropy_coef`       |                `0.01` | Entropy regularization coefficient                 |
| `--grad_clip`          |                `10.0` | Gradient clipping threshold                        |
| `--g_detach`           |                `True` | Detach the Manager goal before Worker optimization |
| `--use_attention`      |                `True` | Enable global–local gated fusion                   |
| `--policy_sharing`     |                `True` | Share network parameters among agents              |

> [!WARNING]
> Several Boolean arguments currently use `type=bool` in `argparse`.
> For reliable Boolean changes, edit `config.py` directly or replace them with `store_true` / `store_false` flags.

## Output Directories

```text
logs/train/                 TensorBoard training logs
models/<environment>/       Model checkpoints
records/<environment>/      Evaluation GIFs
onpolicy/scripts/results/   Baseline on-policy experiment outputs
```

## Project Structure

```text
FeudalMultiAgent/
├── FMANet_mappo.py       # Manager, Worker, intrinsic reward, and full loss
├── dilated_lstm.py       # Dilated LSTM used by the Manager
├── train_mappo.py        # Feudal rollout collection and PPO-style training
├── evaluate.py           # Deterministic evaluation and GIF generation
├── config.py             # Hyperparameters and environment settings
├── utils.py              # Environment, preprocessing, and action utilities
├── logger.py             # TensorBoard logging
├── storage.py            # Rollout storage utilities
├── requirement.txt       # Python dependency snapshot
├── setup.py              # Local onpolicy package installation
├── onpolicy/             # Bundled MAPPO/on-policy baseline code
├── models/               # Saved checkpoints
├── logs/                 # Training logs
└── records/              # Rendered evaluation rollouts
```

## Current Limitations

* The Manager uses centralized information during execution.
* Root-level training currently supports only `simple_spread_v3` and `simple_reference_v3`.
* The recurrent PPO update currently uses full-rollout batches instead of minibatches.
* Returns are computed with discounted bootstrapping rather than GAE.
* Experiment dependencies are tightly pinned and may require adjustment on a different platform.
* The repository is research code and has not yet been packaged as a stable library.

## Acknowledgements

This project builds on ideas and code from:

* **FeUdal Networks for Hierarchical Reinforcement Learning**
* **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**
* The `marlbenchmark/on-policy` implementation
* Multi-Agent Particle Environments and their maintained wrappers

## Citation

When using the hierarchical design, please cite the FeUdal Networks paper:

```bibtex
@inproceedings{vezhnevets2017feudal,
  title     = {FeUdal Networks for Hierarchical Reinforcement Learning},
  author    = {Vezhnevets, Alexander Sasha and Osindero, Simon and Schaul, Tom
               and Heess, Nicolas and Jaderberg, Max and Silver, David
               and Kavukcuoglu, Koray},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year      = {2017}
}
```

For the PPO/MAPPO training foundation:

```bibtex
@inproceedings{yu2022surprising,
  title     = {The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan
               and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

## License

No license file is currently included in this repository. Add an explicit license before distributing or reusing the project.

---

# 中文

## 專案簡介

**FeudalMultiAgent** 是一套用於合作式多智能體強化學習的階層式 Manager–Worker 架構。

高階 **Manager** 整合團隊的全域資訊與個別智能體的局部資訊，並為每一個智能體產生潛在目標；低階 **Worker** 則將自身局部觀測與 Manager 目標融合，輸出動作機率分布。

目前實作包含：

* 階層式 Manager–Worker 決策
* 全域與局部觀測編碼器
* 元素級門控特徵融合
* 用於長時間尺度推理的 Dilated LSTM
* 循環式 Worker 策略
* Manager 與 Worker 雙價值函數
* 以目標一致性為基礎的內在回饋
* PPO／MAPPO 風格的重複更新
* Discrete 與 MultiDiscrete 動作空間
* 智能體之間的參數共享
* TensorBoard 紀錄、模型儲存與軌跡渲染

## 執行階段假設

> [!IMPORTANT]
> 目前 Manager 在動作推論時會接收由所有智能體觀測拼接而成的全域資訊。
> 因此，此版本在執行階段仍會使用集中式資訊，並不是嚴格的 CTDE 分散式執行架構。

整體流程可簡化為：

```text
全域資訊 ──> 全域編碼器 ─────────────────┐
                                           ├─> Manager ─> 潛在目標
局部觀測 ──> 局部編碼器 ─> 門控融合 ─────┘               │
                                                            v
局部觀測 ──> 局部編碼器 ─> 目標門控融合 ─> Worker LSTM
                                                            │
                                                            v
                                                         動作策略
```

Worker critic 另外接收全域特徵，以進行集中式價值估計。

## 網路架構

### 1. 特徵編碼器

全域輸入與局部輸入會分別經過 MLP 投影至潛在特徵空間：

```text
全域輸入 -> percept_G   -> 全域潛在特徵
局部輸入 -> percept_L   -> 局部潛在特徵
局部輸入 -> percept_L4G -> 提供 Manager 使用的局部特徵
全域輸入 -> percept_G4L -> 提供 Worker critic 使用的全域特徵
```

### 2. Manager

Manager 透過元素級門控機制，為每個智能體融合全域與局部潛在特徵：

```text
alpha_i = sigmoid(MLP([z_global, z_local_i]))

z_manager_i = LayerNorm(
    z_global + alpha_i * (z_local_i - z_global)
)
```

融合後的特徵會輸入 Dilated LSTM：

```text
goal_hat_i,t = DilatedLSTM(z_manager_i,t)

goal_i,t = normalize(goal_hat_i,t)
```

Manager 同時估計高階價值：

```text
V_manager_i,t = Critic_manager(goal_hat_i,t)
```

Manager 的學習目標會鼓勵產生的目標方向與未來潛在狀態變化一致：

```text
cos(
    state_i,t+c - state_i,t,
    goal_i,t
)
```

### 3. Worker

Worker 使用另一組元素級門控機制融合局部特徵與 Manager 目標：

```text
beta_i = sigmoid(MLP([z_local_i, goal_i]))

z_worker_i = LayerNorm(
    z_local_i + beta_i * (goal_i - z_local_i)
)
```

融合特徵接著輸入 Worker 循環模組：

```text
u_i,t = LSTM_worker(z_worker_i,t)
```

Worker 支援三種網路模式：

| 模式      | 說明                                     |
| ------- | -------------------------------------- |
| `BASE`  | Worker 循環模組使用局部特徵，並以目標條件化方式計算動作分數。     |
| `GOG`   | 使用觀測–目標門控融合，並以目標條件化方式計算動作分數。           |
| `Ghead` | 使用觀測–目標門控融合，再透過明確的動作輸出層產生策略；目前預設使用此模式。 |

Worker critic 同時使用 Worker 循環特徵與全域特徵：

```text
V_worker_i,t = Critic_worker([u_i,t, z_global])
```

### 4. 內在回饋

Worker 的內在回饋由近期潛在狀態變化與過去 Manager 目標之間的餘弦相似度計算：

```text
r_intrinsic_i,t =
    mean_j cos(
        state_i,t - state_i,t-j,
        goal_i,t-j
    )
```

餘弦相似度會先縮放至 `[0, 1]`，再於 Manager 的時間範圍內取平均。

## 訓練目標

訓練流程分為環境軌跡收集與循環序列重播更新。

軌跡收集階段會儲存動作、回饋、mask、舊策略 log probability、價值估計、循環狀態、Manager 目標與潛在狀態，但不保留原始 computation graph。

每一個 PPO epoch 會重新將整段軌跡輸入目前的網路，重新計算：

* 新策略的動作 log probability
* 策略 entropy
* Manager 價值
* Worker 價值
* Manager 狀態–目標餘弦相似度

### Worker 目標函數

Worker 使用 PPO clipped surrogate objective：

```text
ratio_t = exp(new_log_prob_t - old_log_prob_t)

L_worker_actor =
    -mean(
        min(
            ratio_t * advantage_worker_t,
            clip(ratio_t, 1-epsilon, 1+epsilon)
            * advantage_worker_t
        )
    )
```

### Manager 目標函數

Manager 使用由 Manager advantage 加權的 Feudal cosine objective：

```text
L_manager_actor =
    -mean(
        cosine_t * advantage_manager_t
    )
```

### 價值函數損失

```text
L_manager_value =
    0.5 * mean(
        target_manager - V_manager
    )^2

L_worker_value =
    0.5 * mean(
        target_worker - V_worker
    )^2
```

Worker 的目標值會加入內在回饋：

```text
target_worker =
    extrinsic_return_worker
    + intrinsic_reward_weight * r_intrinsic
```

總損失為：

```text
L_total =
    L_manager_actor
    + L_manager_value
    + L_worker_actor
    + L_worker_value
    - entropy_coefficient * entropy
```

## 支援環境

目前根目錄中的 Feudal 訓練與測試程式實際連接下列合作式 MPE 環境：

| 環境                    | 智能體數量 | 動作空間                                    |
| --------------------- | ----: | --------------------------------------- |
| `simple_spread_v3`    |     3 | 維度為 `5` 的離散移動動作                         |
| `simple_reference_v3` |     2 | 移動與通訊組成的 MultiDiscrete 動作，維度為 `[5, 10]` |

雖然 `onpolicy/` 資料夾內還包含其他環境與基準演算法，但是根目錄的 `train_mappo.py` 與 `evaluate.py` 目前只辨識上述兩個環境。

## 範例軌跡

### Simple Spread

![Simple Spread rollout](records/simple_spread_v3/simple_spread_v3_ATTN_Ghead_comparison5_episode=1600001_seed123456.gif)

### Simple Reference

![Simple Reference rollout](records/simple_reference_v3/simple_reference_v3_ATTN_Ghead_comparison1_episode=120001_seed123456.gif)

## 安裝方式

### 1. 複製儲存庫

```bash
git clone https://github.com/fatgreen/FeudalMultiAgent.git
cd FeudalMultiAgent
```

### 2. 建立 Python 環境

使用 Conda：

```bash
conda create -n feudal-marl python=3.10
conda activate feudal-marl
```

使用虛擬環境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

Linux 或 macOS：

```bash
source .venv/bin/activate
```

### 3. 安裝 PyTorch

請先依照作業系統與 CUDA 版本，安裝相容的 PyTorch。

CPU 安裝範例：

```bash
pip install torch torchvision
```

若使用 CUDA，請使用 PyTorch 官方安裝頁面所提供的對應指令。

### 4. 安裝其餘套件

```bash
pip install -r requirement.txt
pip install -e .
```

> [!NOTE]
> `requirement.txt` 是開發環境的完整套件快照，包含大量固定版本。
> 在不同作業系統或 Python 版本上，部分套件可能需要調整。

## 訓練方式

### Simple Reference

```bash
python train_mappo.py \
    --env_name simple_reference_v3 \
    --run_name reference_ghead
```

Windows PowerShell 或命令提示字元：

```powershell
python train_mappo.py --env_name simple_reference_v3 --run_name reference_ghead
```

### Simple Spread

```bash
python train_mappo.py \
    --env_name simple_spread_v3 \
    --run_name spread_ghead
```

Windows PowerShell 或命令提示字元：

```powershell
python train_mappo.py --env_name simple_spread_v3 --run_name spread_ghead
```

### 指定額外超參數

```bash
python train_mappo.py \
    --env_name simple_reference_v3 \
    --run_name reference_h10 \
    --num_envs 64 \
    --max_total_steps 3000000 \
    --num_steps 25 \
    --ppo_epoch 3 \
    --ppo_clip 0.2 \
    --time_horizon 10 \
    --dilation 10 \
    --lr 0.0007
```

## 模型測試

模型預設從下列位置讀取：

```text
models/<environment>/<model_name>.pt
```

`--load_model` 請輸入不包含 `.pt` 的模型名稱：

```bash
python evaluate.py \
    --env_name simple_reference_v3 \
    --load_model simple_reference_v3_reference_ghead_episode=120001 \
    --test_episodes 8
```

Windows 範例：

```powershell
python evaluate.py --env_name simple_reference_v3 --load_model simple_reference_v3_reference_ghead_episode=120001 --test_episodes 8
```

渲染後的軌跡會儲存在：

```text
records/<environment>/
```

## TensorBoard

訓練紀錄會儲存在：

```text
logs/train/<environment>/
```

啟動 TensorBoard：

```bash
tensorboard --logdir logs/train
```

再開啟終端機中顯示的本機網址。

## 重要設定

主要實驗參數位於 `config.py`。

| 參數                     |                   預設值 | 說明                               |
| ---------------------- | --------------------: | -------------------------------- |
| `--env_name`           | `simple_reference_v3` | 訓練環境                             |
| `--network`            |               `Ghead` | Worker 架構：`BASE`、`GOG` 或 `Ghead` |
| `--num_envs`           |                  `64` | 平行環境數量                           |
| `--max_total_steps`    |             `3000000` | 總環境步數                            |
| `--num_steps`          |                  `25` | 每次更新前收集的 rollout 長度              |
| `--ppo_epoch`          |                   `1` | 每批 rollout 重複更新次數                |
| `--ppo_clip`           |                 `0.5` | PPO clipping 範圍                  |
| `--lr`                 |                `7e-4` | 學習率                              |
| `--time_horizon`       |                  `10` | Manager 時間範圍                     |
| `--dilation`           |                  `10` | Dilated LSTM 循環週期                |
| `--hidden_dim_manager` |                 `256` | Manager 潛在維度                     |
| `--hidden_dim_worker`  |                  `16` | Worker 動作嵌入維度                    |
| `--gamma_m`            |               `0.999` | Manager 折扣因子                     |
| `--gamma_w`            |                `0.99` | Worker 折扣因子                      |
| `--alpha`              |                 `0.5` | 內在回饋係數                           |
| `--entropy_coef`       |                `0.01` | Entropy 正則化係數                    |
| `--grad_clip`          |                `10.0` | 梯度裁切上限                           |
| `--g_detach`           |                `True` | Worker 更新時是否切斷 Manager goal 梯度   |
| `--use_attention`      |                `True` | 是否使用全域–局部門控融合                    |
| `--policy_sharing`     |                `True` | 智能體之間是否共享網路參數                    |

> [!WARNING]
> 目前部分布林參數使用 `argparse` 的 `type=bool`。
> 若要可靠地修改布林設定，建議直接編輯 `config.py`，或將參數改成 `store_true`／`store_false`。

## 輸出資料夾

```text
logs/train/                 TensorBoard 訓練紀錄
models/<environment>/       模型 checkpoint
records/<environment>/      測試軌跡 GIF
onpolicy/scripts/results/   on-policy 基準實驗輸出
```

## 專案結構

```text
FeudalMultiAgent/
├── FMANet_mappo.py       # Manager、Worker、內在回饋與完整 loss
├── dilated_lstm.py       # Manager 使用的 Dilated LSTM
├── train_mappo.py        # 軌跡收集與 PPO 風格訓練
├── evaluate.py           # 確定性測試與 GIF 產生
├── config.py             # 超參數與環境設定
├── utils.py              # 環境、前處理與動作工具
├── logger.py             # TensorBoard 紀錄
├── storage.py            # Rollout 儲存工具
├── requirement.txt       # Python 套件版本快照
├── setup.py              # 本機 onpolicy 套件安裝
├── onpolicy/             # MAPPO／on-policy 基準程式
├── models/               # 已儲存模型
├── logs/                 # 訓練紀錄
└── records/              # 渲染後的測試軌跡
```

## 目前限制

* Manager 在執行階段會使用集中式資訊。
* 根目錄訓練程式目前只支援 `simple_spread_v3` 與 `simple_reference_v3`。
* 循環式 PPO 更新目前使用完整 rollout batch，尚未切分 minibatch。
* Return 使用折扣 bootstrap 計算，而不是 GAE。
* 套件版本固定程度較高，在其他平台上可能需要調整。
* 此專案目前屬於研究程式，尚未整理成穩定的軟體套件。

## 致謝

本專案參考或建立於下列研究與開源程式：

* **FeUdal Networks for Hierarchical Reinforcement Learning**
* **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games**
* `marlbenchmark/on-policy`
* Multi-Agent Particle Environments 與其維護版本

## 引用

若使用本專案的階層式設計，請引用 FeUdal Networks：

```bibtex
@inproceedings{vezhnevets2017feudal,
  title     = {FeUdal Networks for Hierarchical Reinforcement Learning},
  author    = {Vezhnevets, Alexander Sasha and Osindero, Simon and Schaul, Tom
               and Heess, Nicolas and Jaderberg, Max and Silver, David
               and Kavukcuoglu, Koray},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  year      = {2017}
}
```

PPO／MAPPO 訓練基礎請引用：

```bibtex
@inproceedings{yu2022surprising,
  title     = {The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games},
  author    = {Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan
               and Wang, Yu and Bayen, Alexandre and Wu, Yi},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022}
}
```

## 授權

目前儲存庫尚未加入明確的 `LICENSE` 檔案。若要公開散布或供他人重複使用，請先加入授權條款。
