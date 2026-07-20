import os
import json
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24
})
def load_experiment_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    try:
        return np.array(data['mean_reward'])
    except:
        return np.array(data['mean'])

def plot_single_result(mean, std, label):
    episodes = np.arange(len(mean))
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, mean, label=label)
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Performance of {label}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_all_results_together(
    json_paths,
    title="All Experiments Comparison",
    highlight_keywords=None
):
    if highlight_keywords is None:
        highlight_keywords = []

    plt.figure(figsize=(20, 10))

    for path in json_paths:
        mean = load_experiment_results(path)
        episodes = np.arange(len(mean))
        label = os.path.splitext(os.path.basename(path))[0]

        # --- 判斷是否為重點實驗 ---
        is_highlight = any(key in label for key in highlight_keywords)

        if is_highlight:
            plt.plot(
                episodes,
                mean,
                label=label,
                linewidth=2.0,
                alpha=1.0,
                zorder=10
            )
        else:
            plt.plot(
                episodes,
                mean,
                label=label,
                linewidth=1.2,
                alpha=0.8,
                #color="gray",
                zorder=1
            )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 設定資料夾路徑與檔案路徑列表
json_dir = "./1000/change/smoo"
json_paths = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]



# 疊圖比較
plot_all_results_together(
    json_paths,
    title="Webot Experiments Comparison",
    highlight_keywords=["Dyna"]
)