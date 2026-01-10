import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

TARGET_FILE = "data/batch_results_sci.csv"

plt.rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True})


def process_and_plot():
    if not os.path.exists(TARGET_FILE):
        print(f"File not found: {TARGET_FILE}")
        return

    print(f"📂 正在读取全量数据 (使用稀疏采样 1/50 以节省内存)...")

    # [核心修改] 跳行读取！每 50 行取 1 行
    # 这样既能读完 700MB 文件的全貌，又不会爆内存
    try:
        df = pd.read_csv(TARGET_FILE, skiprows=lambda x: x > 0 and x % 50 != 0)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    print(f"✅ 读取完成，当前样本量: {len(df)}")

    # 自动修正列名
    cols = df.columns
    mud_col = next((c for c in cols if 'mud' in c.lower()), 'environment.mud_factor')
    curr_col = next((c for c in cols if 'current' in c.lower()), 'motor_current')
    agent_col = next((c for c in ['agent_id', 'id'] if c in cols), 'agent_id')

    if mud_col not in df.columns:
        # 如果列名不对，强制用最后一列（通常是 mud）
        mud_col = df.columns[-1]

    print(f"🎯 检测到的泥泞度范围: {df[mud_col].unique()}")

    # 1. 画相关性图
    print("🎨 Painting Correlation Plot...")
    plt.figure(figsize=(8, 6))
    # 过滤掉静止的数据
    plot_df = df[df[curr_col].abs() > 0.1]

    sns.lineplot(data=plot_df, x=mud_col, y=curr_col, color='red', label='Mean Load')
    sns.scatterplot(data=plot_df, x=mud_col, y=curr_col, alpha=0.05, color='orange', s=10)

    plt.xlabel("Mud Factor (0.1 - 0.9)")
    plt.ylabel("Actuator Load (A)")
    plt.title("Environmental Coupling Analysis")
    plt.savefig("fig_sci_coupling_corr_fixed.png", dpi=300)
    print("Saved fig_sci_coupling_corr_fixed.png")

    # 2. 画帕累托图
    print("🎨 Painting Pareto Plot...")
    # 聚合
    if 'exp_id' in df.columns:
        summary = df.groupby(['exp_id', agent_col]).agg({
            'time': 'max', 'energy': 'max', mud_col: 'mean'
        }).reset_index()
    else:
        # 如果没有 exp_id，粗略聚合
        summary = df.groupby([mud_col, agent_col]).agg({
            'time': 'max', 'energy': 'max'
        }).reset_index()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=summary, x='time', y='energy',
                    hue=mud_col, palette='viridis', s=100, edgecolor='k')
    plt.title("Pareto Frontier")
    plt.savefig("fig_sci_pareto_frontier_fixed.png", dpi=300)
    print("Saved fig_sci_pareto_frontier_fixed.png")


if __name__ == "__main__":
    process_and_plot()