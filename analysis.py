import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
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


class DeepAnalytics:
    """
    [Data Mining Engine]
    Extracts scientific insights from high-fidelity simulation logs.
    Includes robustness fixes for column name mismatches.
    """

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.df = None

    def load_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not files:
            print("No CSV data found.")
            return False

        # Load all experimental runs
        dfs = []
        for f in files:
            try:
                temp = pd.read_csv(f)
                dfs.append(temp)
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not dfs:
            return False

        self.df = pd.concat(dfs, ignore_index=True)

        # --- [关键修复] 数据标准化 (Data Normalization) ---
        # 1. 统一 Agent ID 列名
        if 'id' in self.df.columns and 'agent_id' not in self.df.columns:
            self.df.rename(columns={'id': 'agent_id'}, inplace=True)

        # 2. 统一 Mud Factor 列名
        if 'mud' in self.df.columns and 'mud_factor' not in self.df.columns:
            self.df.rename(columns={'mud': 'mud_factor'}, inplace=True)
        if 'global_mud_factor' in self.df.columns and 'mud_factor' not in self.df.columns:
            self.df.rename(columns={'global_mud_factor': 'mud_factor'}, inplace=True)

        # 3. 填补缺失的 Experiment ID (兼容 main.py 生成的数据)
        if 'exp_id' not in self.df.columns:
            print("Warning: 'exp_id' column missing. Injecting default value.")
            self.df['exp_id'] = 'Single_Run_Debug'

        # 4. 确保能耗数据存在
        if 'energy' not in self.df.columns:
            self.df['energy'] = 0.0

        print(f"Loaded dataset: {self.df.shape[0]} rows. Columns: {list(self.df.columns)}")
        return True

    def analysis_pareto_efficiency(self):
        """
        [RQ1] System Efficiency vs. Energy Consumption Trade-off.
        Plots the Pareto Frontier across different Mud Factors.
        """
        print("Generating Pareto Frontier...")

        # Group by Episode (Agent + Experiment)
        # 这里的 keys 必须与 load_data 中的标准化列名一致
        summary = self.df.groupby(['exp_id', 'agent_id']).agg({
            'time': 'max',  # Mission Completion Time
            'energy': 'max',  # Total Energy Consumed
            'mud_factor': 'mean'  # Environmental Condition
        }).reset_index()

        plt.figure(figsize=(8, 6))

        # Scatter plot with size/color encoding mud factor
        try:
            scatter = sns.scatterplot(
                data=summary, x='time', y='energy',
                hue='mud_factor', size='mud_factor',
                palette='viridis_r', sizes=(50, 200), alpha=0.8, edgecolor='k'
            )
            plt.legend(title="Mud Factor", bbox_to_anchor=(1.05, 1), loc='upper left')
        except ValueError:
            # Fallback if mud_factor is constant or NaN
            sns.scatterplot(data=summary, x='time', y='energy', s=100)

        plt.title("Pareto Efficiency: Edge Control Performance")
        plt.xlabel("Mission Duration (s) [Efficiency]")
        plt.ylabel("Total Energy (J) [Cost]")
        plt.tight_layout()
        plt.savefig("fig_sci_pareto_frontier.png", dpi=300)
        print("Saved fig_sci_pareto_frontier.png")

    def analysis_vibration_spectrum(self):
        """
        [RQ2] Multi-Body Dynamics Validation via Frequency Domain.
        Computes PSD of Motor Current to show low-frequency longitudinal oscillations.
        """
        print("Generating Vibration PSD...")

        plt.figure(figsize=(8, 5))

        agents = self.df['agent_id'].unique()
        for agent in agents:
            # Filter trajectory
            traj = self.df[self.df['agent_id'] == agent]
            if len(traj) < 100: continue

            # Analyze Coupler Force or Acceleration proxy (Current)
            if 'motor_current' in traj.columns:
                signal = traj['motor_current'].values
            elif 'current' in traj.columns:  # 兼容不同命名
                signal = traj['current'].values
            else:
                continue

            # Remove DC component
            signal = signal - np.mean(signal)

            # Welch's Method for PSD
            fs = 10.0  # Sampling freq (dt=0.1)
            # 兼容旧版本 scipy 语法
            try:
                f, Pxx = welch(signal, fs, nperseg=min(256, len(signal)))
                plt.semilogy(f, Pxx, label=f"{agent}")
            except Exception as e:
                print(f"Skipping PSD for {agent}: {e}")

        plt.title("Power Spectral Density (PSD) of Motor Current")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (A²/Hz)")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("fig_sci_vibration_psd.png", dpi=300)
        print("Saved fig_sci_vibration_psd.png")

    def analysis_coupling_correlation(self):
        """
        [RQ3] Electro-Mechanical Coupling Analysis.
        Quantifies how 'Mud Factor' affects 'Motor Efficiency'.
        """
        print("Generating Coupling Correlation...")

        # Calculate instant power
        self.df['power_inst'] = self.df['energy'].diff().fillna(0) / 0.1  # Watts

        # 兼容列名 velocity 或 vel
        vel_col = 'velocity' if 'velocity' in self.df.columns else 'vel'
        curr_col = 'motor_current' if 'motor_current' in self.df.columns else 'current'

        if vel_col not in self.df.columns or curr_col not in self.df.columns:
            return

        # Select high-speed cruising phase
        cruise_data = self.df[(self.df[vel_col] > 1.0) & (self.df[vel_col] < 5.0)]

        if cruise_data.empty:
            print("Not enough cruise data for correlation analysis.")
            return

        plt.figure(figsize=(8, 6))

        # 如果 mud_factor 是常数（单次运行），则无法做相关性分析，只画散点
        if cruise_data['mud_factor'].nunique() <= 1:
            sns.scatterplot(data=cruise_data, x='mud_factor', y=curr_col, alpha=0.5)
            plt.title("Actuator Load Distribution (Single Mud Factor)")
        else:
            sns.regplot(data=cruise_data, x='mud_factor', y=curr_col,
                        x_bins=10, order=2, color='darkred', line_kws={'linestyle': '--'})
            r_val, p_val = pearsonr(cruise_data['mud_factor'], cruise_data[curr_col])
            plt.title(f"Environmental Impact (Pearson r={r_val:.2f})")

        plt.xlabel("Mud Factor (Soil Plasticity)")
        plt.ylabel("Average Motor Current (A)")
        plt.tight_layout()
        plt.savefig("fig_sci_coupling_corr.png", dpi=300)
        print("Saved fig_sci_coupling_corr.png")

if __name__ == "__main__":
    analyzer = DeepAnalytics()
    if analyzer.load_data():
        analyzer.analysis_pareto_efficiency()
        analyzer.analysis_vibration_spectrum()
        analyzer.analysis_coupling_correlation()
        print("SCI-Grade Analysis Complete. Figures saved.")