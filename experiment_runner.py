import pandas as pd
import os
import logging
import time
from tqdm import tqdm

# Import Core Modules
from config_loader import ConfigLoader
from map_core import GridMap
from vehicle import VehicleAgent

logging.basicConfig(level=logging.WARNING)  # 减少日志输出，提高速度


class HeadlessRunner:
    """
    [Experiment Automation]
    Runs batch simulations for statistical significance (Monte Carlo).
    """

    def __init__(self, config):
        self.cfg = config
        self.env = config['environment']
        self.grid = GridMap(config)

        # 收集道岔代理
        self.infra_agents = {
            nid: node.agent
            for nid, node in self.grid.nodes.items()
            if node.agent is not None
        }

        self.vehicles = []
        self._init_pop()
        self.logs = []

    def _init_pop(self):
        # 批量生成车辆
        # 示例：1个重型车，1个侦察车
        scenarios = [
            ('Heavy_Hauler', 'Start_1', 0.0),
            ('Fast_Scout', 'Start_2', 10.0)
        ]

        for i, (v_type, start_node, delay) in enumerate(scenarios):
            v = VehicleAgent(
                agent_id=f"V_{i}_{v_type}",
                vehicle_type_cfg=self.cfg['vehicle_types'][v_type],  # [修复] 正确传参
                env_config=self.env,
                start_node=start_node,
                map_graph=self.grid,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(v)

        # Link V2V
        for veh in self.vehicles: veh.all_vehicles = self.vehicles

    def run_episode(self):
        t_max = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']
        t = 0.0

        # 纯计算循环，无 GUI，速度极快
        while t < t_max:
            # 1. Update Infrastructure
            self.grid.update_infrastructure(dt, t)

            # 2. Update Vehicles
            for v in self.vehicles:
                log = v.step(dt, t)
                if log:
                    log.update({
                        'time': t,
                        'exp_id': self.cfg['meta']['experiment_id'],
                        'mud': self.env['mud_factor']
                    })
                    self.logs.append(log)

            t += dt

        return pd.DataFrame(self.logs)


if __name__ == "__main__":
    # [SCI 核心配置] 定义扫描计划
    # 1. 泥泞度：从 0.1 到 0.9，每隔 0.1 测一次 -> 生成 Actuator Load 横向分布图
    # 2. 车辆类型：覆盖重载车和侦察车 -> 生成 Pareto 异构对比
    sweep_plan = {
        'environment.mud_factor': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # 如果你想跑得快一点，可以注释掉下面这行（只跑默认车辆）
        # 但为了 Pareto 图好看，建议保留
        'vehicle_types.Heavy_Hauler.pid.kp': [2000]
    }

    if not os.path.exists("config.yaml"):
        print("Error: config.yaml missing")
        exit()

    all_results = []
    print("Starting Batch Simulation for SCI Analysis...")
    print("(This process simulates multiple episodes, please wait...)")

    # 生成配置矩阵
    configs = list(ConfigLoader.generate_sweep("config.yaml", sweep_plan))

    # 使用 tqdm 显示进度条
    for cfg in tqdm(configs):
        runner = HeadlessRunner(cfg)
        df = runner.run_episode()
        all_results.append(df)

    # 合并并保存
    if all_results:
        final_df = pd.concat(all_results)
        # 保存为 analysis-optimized.py 能识别的文件名格式
        final_df.to_csv("data/batch_results_sci.csv", index=False)
        print(f"Batch Simulation Complete. Generated {len(final_df)} data points.")
    else:
        print("No results generated.")