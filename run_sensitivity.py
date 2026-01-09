# [New File: run_sensitivity.py]
import numpy as np
import pandas as pd
from config_loader import ConfigLoader
from main import DecentralizedSimulation
import matplotlib.pyplot as plt


def run_sweep():
    base_cfg = "config.yaml"
    # 定义 rho = we / wt 的扫描范围 (对数刻度: 0.1 到 10)
    rhos = np.logspace(-1, 1, 10)
    results = []

    print("Starting Sensitivity Analysis (Rho = We/Wt)...")

    for rho in rhos:
        # 动态修改配置
        cfg = ConfigLoader.load(base_cfg)

        # 设定权重 (假设 wt 固定为 1.0)
        # 注意：你需要修改 router.py 让他能接受这些参数（见补充4）
        cfg['algorithms']['weights'] = {'alpha_t': 1.0, 'beta_e': rho, 'gamma_r': 100.0}

        print(f"Running for rho = {rho:.2f}...")
        sim = DecentralizedSimulation()
        sim.cfg = cfg  # 注入修改后的配置
        # 重新初始化路由模块以应用新权重
        # (这步需要在 main.py 的 __init__ 中支持传入 cfg，或者在此处手动 patch)
        for nid, node in sim.map.nodes.items():
            if hasattr(node.agent, 'router'):  # 假设 SwitchAgent 有路由模块
                node.agent.router.evaluator.alpha_t = 1.0
                node.agent.router.evaluator.beta_e = rho

        # 运行仿真 (加速模式，无GUI)
        sim.cfg['simulation']['visualization'] = False
        sim.run()

        # 收集数据
        df = pd.DataFrame(sim.logs)
        avg_energy = df['energy'].max()  # 总能耗
        mission_time = df['time'].max()  # 完赛时间

        results.append({
            'rho': rho,
            'energy': avg_energy,
            'time': mission_time
        })

    # 保存结果供画图
    res_df = pd.DataFrame(results)
    res_df.to_csv("data/sensitivity_results.csv", index=False)
    print("Sensitivity Analysis Complete.")


if __name__ == "__main__":
    run_sweep()