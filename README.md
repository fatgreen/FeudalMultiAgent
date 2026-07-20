# FeudalMultiAgent

**A Hierarchical Manager–Worker Framework for Cooperative Multi-Agent Reinforcement Learning**

[English](#english) | [中文](#中文)

---

# English

## Introduction

**FeudalMultiAgent** is a research implementation of hierarchical cooperative multi-agent reinforcement learning.

The framework uses a high-level **Manager** to generate latent goals and a low-level **Worker** to select actions according to local observations and the generated goals. The training procedure combines hierarchical reinforcement learning with PPO/MAPPO-style policy updates.

This repository is developed for experiments on cooperative multi-agent tasks and is based partly on the open-source MAPPO/on-policy implementation.

## Main Features

* Hierarchical Manager–Worker policy
* Recurrent neural networks for temporal information
* Goal-based intrinsic rewards
* PPO/MAPPO-style repeated policy updates
* Shared policy parameters among agents
* TensorBoard logging and model evaluation
* Support for Discrete and MultiDiscrete action spaces

## Supported Environments

| Environment           | Agents | Description                                         |
| --------------------- | -----: | --------------------------------------------------- |
| `simple_spread_v3`    |      3 | Agents cooperate to cover multiple landmarks        |
| `simple_reference_v3` |      2 | Agents cooperate through movement and communication |

## Experimental Results

The following figures compare FeudalMultiAgent with other multi-agent reinforcement learning methods. Solid lines show the mean performance, while the shaded regions represent the standard deviation across different random seeds.

### Simple Spread

#### Performance Comparison

<p align="center">
  <img src="img/simple_spread_v3/simplespread_mean_std_shaded.png" alt="Simple Spread performance comparison" width="900">
</p>

#### Evaluation Rollout

<p align="center">
  <img src="records/simple_spread_v3/simple_spread_v3_ATTN_Ghead_comparison5_episode=1600001_seed123456.gif" alt="Simple Spread evaluation rollout" width="500">
</p>

### Simple Reference

#### Performance Comparison

<p align="center">
  <img src="img/simple_reference_v3/simplereference_mean_std_shaded.png" alt="Simple Reference performance comparison" width="900">
</p>

#### Evaluation Rollout

<p align="center">
  <img src="records/simple_reference_v3/simple_reference_v3_ATTN_Ghead_comparison1_episode=120001_seed123456.gif" alt="Simple Reference evaluation rollout" width="500">
</p>

## Installation

Clone the repository:

```bash
git clone https://github.com/fatgreen/FeudalMultiAgent.git
cd FeudalMultiAgent
```

Create and activate a Python environment:

```bash
conda create -n feudal-marl python=3.10
conda activate feudal-marl
```

Install PyTorch according to your operating system and CUDA version. Then install the remaining dependencies:

```bash
pip install -r requirement.txt
pip install -e .
pip install seaborn
pip install pyglet==1.5.27
```

## Training

Train on Simple Reference:

```bash
python train_mappo.py --env_name simple_reference_v3
```

Train on Simple Spread:

```bash
python train_mappo.py --env_name simple_spread_v3
```

The main experiment settings can be modified in `config.py` or passed through command-line arguments.

## Evaluation

Models are loaded from:

```text
models/<environment>/<model_name>.pt
```

Example:

```bash
python evaluate.py --env_name simple_reference_v3 --load_model <model_name>
```

Rendered evaluation results are saved under:

```text
records/<environment>/
```

## TensorBoard

Training logs are stored under:

```text
logs/train/<environment>/
```

Start TensorBoard with:

```bash
tensorboard --logdir logs/train/<environment>
```

Example:

```bash
tensorboard --logdir logs/train/simple_reference_v3
```

## Main Files

```text
FMANet_mappo.py   Manager–Worker network and training losses
train_mappo.py    Training program
evaluate.py       Model evaluation and GIF generation
config.py         Experiment settings
dilated_lstm.py   Dilated LSTM module
utils.py          Environment and preprocessing utilities
logger.py         TensorBoard logger
onpolicy/         MAPPO/on-policy source code
```

## Acknowledgements

This project is inspired by:

* FeUdal Networks for Hierarchical Reinforcement Learning
* The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
* The `marlbenchmark/on-policy` repository
* Multi-Agent Particle Environments

---

# 中文

## 專案介紹

**FeudalMultiAgent** 是一套用於合作式多智能體強化學習的階層式 Manager–Worker 研究架構。

高階 **Manager** 負責產生潛在目標，低階 **Worker** 則根據局部觀測與目標選擇動作。訓練方式結合階層式強化學習以及 PPO／MAPPO 風格的策略更新。

本專案主要用於合作式多智能體環境實驗，部分程式建立於開源 MAPPO／on-policy 實作之上。

## 主要特色

* 階層式 Manager–Worker 策略
* 使用循環神經網路處理時間資訊
* 以目標為基礎的內在回饋
* PPO／MAPPO 風格的重複策略更新
* 智能體之間共享策略參數
* TensorBoard 訓練紀錄與模型測試
* 支援 Discrete 與 MultiDiscrete 動作空間

## 支援環境

| 環境                    | 智能體數量 | 說明               |
| --------------------- | ----: | ---------------- |
| `simple_spread_v3`    |     3 | 多個智能體合作覆蓋不同地標    |
| `simple_reference_v3` |     2 | 智能體透過移動與通訊完成合作任務 |

## 實驗成果

下列結果比較 FeudalMultiAgent 與其他多智能體強化學習方法的訓練表現。實線表示不同隨機種子實驗結果的平均值，陰影區域表示標準差。

### Simple Spread

#### 方法比較

<p align="center">
  <img src="img/simple_spread_v3/simplespread_mean_std_shaded.png" alt="Simple Spread 方法比較" width="900">
</p>

#### 測試結果

<p align="center">
  <img src="records/simple_spread_v3/simple_spread_v3_ATTN_Ghead_comparison5_episode=1600001_seed123456.gif" alt="Simple Spread 測試結果" width="500">
</p>

### Simple Reference

#### 方法比較

<p align="center">
  <img src="img/simple_reference_v3/simplereference_mean_std_shaded.png" alt="Simple Reference 方法比較" width="900">
</p>

#### 測試結果

<p align="center">
  <img src="records/simple_reference_v3/simple_reference_v3_ATTN_Ghead_comparison1_episode=120001_seed123456.gif" alt="Simple Reference 測試結果" width="500">
</p>

## 安裝方式

複製專案：

```bash
git clone https://github.com/fatgreen/FeudalMultiAgent.git
cd FeudalMultiAgent
```

建立並啟用 Python 環境：

```bash
conda create -n feudal-marl python=3.10
conda activate feudal-marl
```

請先依照作業系統與 CUDA 版本安裝 PyTorch，再安裝其他套件：

```bash
pip install -r requirement.txt
pip install -e .
pip install seaborn
pip install pyglet==1.5.27
```

## 模型訓練

訓練 Simple Reference：

```bash
python train_mappo.py --env_name simple_reference_v3
```

訓練 Simple Spread：

```bash
python train_mappo.py --env_name simple_spread_v3
```

主要實驗參數可在 `config.py` 中修改，也可以透過命令列指定。

## 模型測試

模型預設從以下路徑載入：

```text
models/<environment>/<model_name>.pt
```

執行範例：

```bash
python evaluate.py --env_name simple_reference_v3 --load_model <model_name>
```

測試產生的 GIF 會儲存在：

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
tensorboard --logdir logs/train/<environment>
```

範例：

```bash
tensorboard --logdir logs/train/simple_reference_v3
```

## 主要檔案

```text
FMANet_mappo.py   Manager–Worker 網路與訓練損失
train_mappo.py    模型訓練程式
evaluate.py       模型測試與 GIF 產生
config.py         實驗參數設定
dilated_lstm.py   Dilated LSTM 模組
utils.py          環境與資料前處理工具
logger.py         TensorBoard 紀錄工具
onpolicy/         MAPPO／on-policy 原始程式
```

## 致謝

本專案參考下列研究與開源程式：

* FeUdal Networks for Hierarchical Reinforcement Learning
* The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
* `marlbenchmark/on-policy`
* Multi-Agent Particle Environments
