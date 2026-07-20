import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# =========================================================
# 1. 設定資料根目錄
# =========================================================
ROOT_DIR = Path(r"D:\!!Fatgreen\!Research\report\!!csv\SimpleReference")

# 你的 7 個架構資料夾名稱
method_order = [
    "COMA",
    "Feudal",
    "FHMA",
    "IPPO",
    "MADDPG",
    "MAPPO",
    "QMIX",
]

# 用最後幾個點的平均 reward 來排名
# 例如 10 代表用最後 10 個 Step 的平均 reward 排名
RANK_LAST_N = 10
RANK_RATIO = 0.5  # 用前多少比例的 training steps 排名，0.5 代表用左半邊，也就是前 50% 訓練區間
# 輸出資料夾
OUT_DIR = ROOT_DIR / "summary_results"
OUT_DIR.mkdir(exist_ok=True)

# 圖片與標題使用資料夾名稱
ENV_NAME = ROOT_DIR.name


# =========================================================
# 2. 自動讀取每個方法資料夾底下的 csv
# =========================================================
groups = {}

for method in method_order:
    method_dir = ROOT_DIR / method

    if not method_dir.exists():
        print(f"[Warning] Folder not found: {method_dir}")
        continue

    csv_files = sorted(method_dir.glob("*.csv"))

    if len(csv_files) == 0:
        print(f"[Warning] No csv files found in: {method_dir}")
        continue

    groups[method] = csv_files
    print(f"{method}: found {len(csv_files)} csv files")

if len(groups) == 0:
    raise RuntimeError("No csv files found. Please check ROOT_DIR and folder names.")


# =========================================================
# 3. 計算每個 Step 的 mean / std / var
# =========================================================
def summarize_group(file_list):
    all_dfs = []

    for path in file_list:
        df = pd.read_csv(path)

        # 避免欄位名稱前後有空白
        df.columns = df.columns.str.strip()

        # TensorBoard 下載的 csv 通常會有 Wall time, Step, Value
        if "Step" not in df.columns or "Value" not in df.columns:
            raise ValueError(
                f"{path.name} 欄位不符合格式，目前欄位為：{list(df.columns)}"
            )

        df = df[["Step", "Value"]].copy()
        df["source_file"] = path.stem
        all_dfs.append(df)

    data = pd.concat(all_dfs, ignore_index=True)

    summary = (
        data.groupby("Step", as_index=False)
        .agg(
            mean_value=("Value", "mean"),
            std_value=("Value", "std"),
            var_value=("Value", "var"),
            count=("Value", "count"),
        )
        .sort_values("Step")
    )

    # 如果某些 Step 只有一筆資料，std / var 會是 NaN，補成 0
    summary["std_value"] = summary["std_value"].fillna(0)
    summary["var_value"] = summary["var_value"].fillna(0)

    return summary


# =========================================================
# 4. 對所有方法做統計並輸出 summary csv
# =========================================================
summaries = {}

for method_name, file_list in groups.items():
    summaries[method_name] = summarize_group(file_list)

    out_csv = OUT_DIR / f"{method_name}_summary_mean_std_variance.csv"
    summaries[method_name].to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Saved: {out_csv}")


# =========================================================
# 5. 根據 reward 表現高到低排序 SS
# =========================================================
method_rank = {}

for method_name, summary in summaries.items():
    last_rewards = summary["mean_value"].tail(RANK_LAST_N)
    method_rank[method_name] = last_rewards.mean()

# reward 高到低排序
plot_order = sorted(
    summaries.keys(),
    key=lambda m: method_rank[m],
    reverse=True
)

if "FHMA" in plot_order:
    plot_order.remove("FHMA")
    plot_order.insert(0, "FHMA")

print("\nMethod ranking by final reward:")
for i, method_name in enumerate(plot_order, start=1):
    print(f"{i}. {method_name}: {method_rank[method_name]:.4f}")


# =========================================================
# 5. 根據左半邊收斂速度排序 SR
# =========================================================

# 使用前多少比例的 training steps 排名
# 0.5 代表用左半邊，也就是前 50% 訓練區間
# RANK_RATIO = 0.5

# method_rank = {}

# for method_name, summary in summaries.items():
#     max_step = summary["Step"].max()
#     rank_step = max_step * RANK_RATIO

#     # 取左半邊訓練區間
#     early_region = summary[summary["Step"] <= rank_step]

#     # reward 越高代表越快收斂、表現越好
#     method_rank[method_name] = early_region["mean_value"].mean()

# # 根據左半邊平均 reward 高到低排序
# plot_order = sorted(
#     summaries.keys(),
#     key=lambda m: method_rank[m],
#     reverse=True
# )

# print("\nMethod ranking by early convergence speed:")
# for i, method_name in enumerate(plot_order, start=1):
#     print(f"{i}. {method_name}: {method_rank[method_name]:.4f}")


# =========================================================
# 固定每個方法的顏色，避免排序後顏色改變
# =========================================================
color_map = {
    "FHMA": "tab:blue",
    "MADDPG": "tab:orange",
    "MAPPO": "tab:green",
    "IPPO": "tab:red",
    "QMIX": "tab:purple",
    "Feudal": "tab:brown",
    "COMA": "tab:pink",
}

# =========================================================
# 5.1 輸出 PPT 表格用數據
# =========================================================

table_rows = []

for method_name in plot_order:
    summary = summaries[method_name]

    max_step = summary["Step"].max()
    rank_step = max_step * RANK_RATIO

    # 前 50% 訓練區間，代表前期收斂表現
    early_region = summary[summary["Step"] <= rank_step]

    # 最後 N 個點，代表最終收斂表現
    final_region = summary.tail(RANK_LAST_N)

    early_reward = early_region["mean_value"].mean()
    final_reward = final_region["mean_value"].mean()
    final_std = final_region["std_value"].mean()
    final_var = final_region["var_value"].mean()

    table_rows.append({
        "Method": method_name,
        "Early Reward": early_reward,
        "Final Reward": final_reward,
        "Final STD": final_std,
        "Final Variance": final_var,
    })

metrics_df = pd.DataFrame(table_rows)

# 排名照目前 plot_order，也就是左半邊收斂速度排序
metrics_df.insert(0, "Rank", range(1, len(metrics_df) + 1))

# 印出比較好複製到 PPT 的格式
print("\nPPT table data:")
print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# 也輸出成 csv，方便直接複製到 PPT 或 Excel
table_csv = OUT_DIR / f"{ENV_NAME.lower()}_ppt_table_metrics.csv"
metrics_df.to_csv(table_csv, index=False, encoding="utf-8-sig")

print(f"\nSaved PPT table csv: {table_csv}")



# =========================================================
# 6. 畫 mean ± STD 圖
# =========================================================
plt.figure(figsize=(12, 6))

for method_name in plot_order:
    summary = summaries[method_name]

    x = summary["Step"].to_numpy()
    mean = summary["mean_value"].to_numpy()
    std = summary["std_value"].to_numpy()

    plt.plot(x, mean, label=method_name, color=color_map.get(method_name, None))
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

plt.xlabel("Training Steps")
plt.ylabel("Total Reward")
plt.title(f"{ENV_NAME}: Total Reward Mean ± STD")
plt.grid(True, alpha=0.3)

# 右下角 method 標示，順序為 reward 高到低
plt.legend(loc="lower right")

plt.tight_layout()

mean_std_fig = OUT_DIR / f"{ENV_NAME.lower()}_mean_std_shaded.png"
plt.savefig(mean_std_fig, dpi=300)
plt.show()

print(f"Saved figure: {mean_std_fig}")


# =========================================================
# 7. 畫 variance 線圖
# =========================================================
plt.figure(figsize=(12, 6))

for method_name in plot_order:
    summary = summaries[method_name]

    plt.plot(
        summary["Step"].to_numpy(),
        summary["var_value"].to_numpy(),
        label=method_name,
        color=color_map.get(method_name, None)
    )

plt.xlabel("Training Steps")
plt.ylabel("Variance")
plt.title(f"{ENV_NAME}: Variance of Total Reward")
plt.grid(True, alpha=0.3)

# 右下角 method 標示，順序一樣照 reward 高到低
plt.legend(loc="lower right")

plt.tight_layout()

var_fig = OUT_DIR / f"{ENV_NAME.lower()}_variance_line.png"
plt.savefig(var_fig, dpi=300)
plt.show()

print(f"Saved figure: {var_fig}")