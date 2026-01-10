import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import welch
from scipy.stats import pearsonr

# 配置 SCI 绘图风格
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 1.5
})


class BigDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        # 数据容器
        self.pareto_records = []
        self.corr_records = []
        self.psd_samples = {}  # key: agent_id, value: list of currents
        self.agents_found = set()

    def process_file(self):
        print(f"Streaming large file: {self.file_path} ...")

        # [核心优化] 分块读取，每次只读 10万行，避免内存爆炸
        chunk_size = 100000
        reader = pd.read_csv(self.file_path, chunksize=chunk_size)

        processed_rows = 0

        for chunk in reader:
            # 1. 标准化列名
            if 'id' in chunk.columns and 'agent_id' not in chunk.columns:
                chunk.rename(columns={'id': 'agent_id'}, inplace=True)
            if 'mud' in chunk.columns and 'mud_factor' not in chunk.columns:
                chunk.rename(columns={'mud': 'mud_factor'}, inplace=True)
            if 'global_mud_factor' in chunk.columns:
                chunk.rename(columns={'global_mud_factor': 'mud_factor'}, inplace=True)
            if 'motor_current' not in chunk.columns and 'current' in chunk.columns:
                chunk.rename(columns={'current': 'motor_current'}, inplace=True)

            # 2. 提取 Pareto 数据 (聚合每个 Episode 的最终结果)
            # 我们假设每个 chunk 可能包含某些 episode 的中间数据，
            # 最准确的方法是最后聚合，但为了省内存，我们可以提取每个chunk里的最大值作为候选
            # 更精准的方法：只保留 chunk 里 time 最大的那些行（如果数据是按时间排序的）
            if 'exp_id' in chunk.columns:
                # 提取每个 exp_id + agent_id 的摘要
                summary = chunk.groupby(['exp_id', 'agent_id', 'mud_factor']).agg({
                    'time': 'max',
                    'energy': 'max'
                }).reset_index()
                self.pareto_records.append(summary)

            # 3. 提取 Correlation 数据 (降采样)
            # 800MB 数据太多了，我们只需要随机抽取 1% 的点画散点图就够了
            sample_chunk = chunk.sample(frac=0.01, random_state=42)
            self.corr_records.append(sample_chunk[['mud_factor', 'motor_current', 'vel']])

            # 4. 提取 PSD 数据 (只取头几万个点)
            # 我们只需要每个车辆的一段连续数据即可
            for agent in chunk['agent_id'].unique():
                if agent not in self.psd_samples:
                    # 找到该 agent 的数据
                    agent_data = chunk[chunk['agent_id'] == agent]['motor_current'].values
                    # 只需要存 5000 个点就够做 FFT 了
                    if len(agent_data) > 100:
                        self.psd_samples[agent] = agent_data[:10000]  # 截取前1万个点

            processed_rows += len(chunk)
            print(f"Processed {processed_rows} rows...", end='\r')

        print("\nFile processing complete.")

    def plot_pareto(self):
        print("Generating Pareto Plot...")
        # 合并所有 chunk 的摘要
        full_df = pd.concat(self.pareto_records)
        # 再次聚合，因为一个 episode 可能跨越了多个 chunk
        final_summary = full_df.groupby(['exp_id', 'agent_id']).agg({
            'time': 'max',
            'energy': 'max',
            'mud_factor': 'mean'
        }).reset_index()

        plt.figure(figsize=(8, 6))
        try:
            sns.scatterplot(
                data=final_summary, x='time', y='energy',
                hue='mud_factor', size='mud_factor',
                palette='viridis_r', sizes=(50, 200), alpha=0.8, edgecolor='k'
            )
            plt.legend(title="Mud Factor", bbox_to_anchor=(1.05, 1), loc='upper left')
        except Exception as e:
            print(f"Pareto plot simplified due to error: {e}")
            sns.scatterplot(data=final_summary, x='time', y='energy')

        plt.title("Pareto Efficiency: Edge Control Performance")
        plt.xlabel("Mission Duration (s)")
        plt.ylabel("Total Energy (J)")
        plt.tight_layout()
        plt.savefig("fig_sci_pareto_frontier.png", dpi=300)
        print("Saved fig_sci_pareto_frontier.png")

    def plot_correlation(self):
        print("Generating Correlation Plot...")
        full_sample = pd.concat(self.corr_records)

        # 过滤掉静止状态的数据 (速度太小的不算负载)
        if 'vel' in full_sample.columns:
            cruise_data = full_sample[(full_sample['vel'].abs() > 0.1)]
        else:
            cruise_data = full_sample

        plt.figure(figsize=(8, 6))

        # 如果泥泞度有变化，画相关性回归图
        if cruise_data['mud_factor'].nunique() > 1:
            sns.regplot(data=cruise_data, x='mud_factor', y='motor_current',
                        x_bins=10, order=2, color='darkred',
                        line_kws={'linestyle': '--', 'color': 'black'},
                        scatter_kws={'alpha': 0.1})  # 透明度调高，因为点多

            try:
                r_val, _ = pearsonr(cruise_data['mud_factor'], cruise_data['motor_current'])
                plt.title(f"Environmental Impact (Pearson r={r_val:.2f})")
            except:
                pass
        else:
            sns.scatterplot(data=cruise_data, x='mud_factor', y='motor_current', alpha=0.1)
            plt.title("Actuator Load Distribution (Single Mud Factor)")

        plt.xlabel("Mud Factor")
        plt.ylabel("Motor Current (A)")
        plt.tight_layout()
        plt.savefig("fig_sci_coupling_corr.png", dpi=300)
        print("Saved fig_sci_coupling_corr.png")

    def plot_psd(self):
        print("Generating PSD Plot...")
        plt.figure(figsize=(8, 5))

        for agent, signal in self.psd_samples.items():
            # 简单的清洗
            signal = np.nan_to_num(signal)
            signal = signal - np.mean(signal)

            if len(signal) < 100: continue

            fs = 100.0  # 假设之前是 0.01s 一个点 (100Hz)
            # 如果之前是 0.1s 一个点，这里改 10.0

            try:
                f, Pxx = welch(signal, fs, nperseg=min(1024, len(signal)))
                plt.semilogy(f, Pxx, label=str(agent))
            except:
                pass

        plt.title("Power Spectral Density (PSD) of Motor Current")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (A²/Hz)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("fig_sci_vibration_psd.png", dpi=300)
        print("Saved fig_sci_vibration_psd.png")


if __name__ == "__main__":
    # 自动寻找 data 目录下最大的 csv 文件
    data_dir = "data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in data/")
        exit()

    # 找最大的那个文件
    largest_file = max([os.path.join(data_dir, f) for f in files], key=os.path.getsize)
    print(f"Analyzing largest file: {largest_file}")

    analyzer = BigDataAnalyzer(largest_file)
    analyzer.process_file()

    analyzer.plot_pareto()
    analyzer.plot_correlation()
    analyzer.plot_psd()
    print("All plots generated successfully!")