import logging
import time
import os
import pandas as pd
from datetime import datetime

# 引入核心模块
from config_loader import ConfigLoader
from map_core import GridMap
from vehicle import VehicleAgent
from visualization import SimVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger("MainLoop")


class DecentralizedSimulation:
    """
    [Main Orchestrator]
    Discrete-Event Simulation for Distributed Edge Control System.
    No central controller logic here - only physics stepping and time management.
    """

    def __init__(self, config_path="config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config {config_path} not found")

        self.cfg = ConfigLoader.load(config_path)
        self.env_cfg = self.cfg['environment']

        # 1. 初始化异构环境与边缘设施
        logger.info("Initializing Heterogeneous Map & Edge Infrastructure...")
        self.map = GridMap(self.cfg)

        # 收集所有的道岔代理 (Infrastructure Agents)
        # 格式: {node_id: EdgeSwitchAgent}
        self.infra_agents = {
            nid: node.agent
            for nid, node in self.map.nodes.items()
            if node.agent is not None
        }
        logger.info(f"Deployed {len(self.infra_agents)} Edge Switch Agents.")

        # 2. 初始化车辆智能体 (Vehicle Agents)
        logger.info("Deploying Autonomous Vehicles...")
        self.vehicles = []

        # --- V1: Heavy Hauler (Logistics) ---
        v1_type = "Heavy_Hauler"
        v1 = VehicleAgent(
            agent_id="HV_Hauler_01",
            vehicle_type_cfg=self.cfg['vehicle_types'][v1_type],  # 直接传入该车型的配置字典
            env_config=self.env_cfg,
            start_node="Start_1",
            map_graph=self.map,
            infra_agents=self.infra_agents  # 传入字典索引，而非列表
        )
        # 手动设置启动延时 (如果 VehicleAgent 移除了该参数，我们可以通过修改内部状态模拟)
        # v1.start_delay = 0.0 # VehicleAgent 最新版使用内部逻辑触发
        self.vehicles.append(v1)

        # --- V2: Fast Scout (Inspection) ---
        v2_type = "Fast_Scout"
        v2 = VehicleAgent(
            agent_id="FS_Scout_02",
            vehicle_type_cfg=self.cfg['vehicle_types'][v2_type],
            env_config=self.env_cfg,
            start_node="Start_2",
            map_graph=self.map,
            infra_agents=self.infra_agents
        )
        self.vehicles.append(v2)

        # [SCI Depth] 注入 V2V 引用 (如果 vehicle.py 中需要进行车辆间通信/防撞)
        # 即使 __init__ 没传，也可以通过属性注入
        for v in self.vehicles:
            v.all_vehicles = self.vehicles

        # 3. 数据记录与可视化
        self.logs = []
        # 初始化可视化 (传递引用)
        self.viz = SimVisualizer(self.map, self.vehicles, self.cfg)

    def step(self, t, dt):
        """
        [Parallel Execution Emulation]
        Strictly separated phases to simulate distributed system.
        """
        # Phase 1: Infrastructure Update (Physics & Local Logic)
        self.map.update_infrastructure(dt, t)

        # Phase 2: Vehicle Agents Update (Sense -> Plan -> Act)
        step_data = []
        for v in self.vehicles:
            # 模拟：车辆独立运行步进
            log = v.step(dt, t)

            # 注入 Ground Truth (用于论文分析对比，车辆自己不知道这些)
            if log:
                log.update({
                    'time': t,
                    'sim_step': int(t / dt),
                    'global_mud_factor': self.env_cfg['mud_factor']
                })
                step_data.append(log)

        self.logs.extend(step_data)

    def run(self):
        duration = self.cfg['simulation']['duration']
        dt = self.cfg['simulation']['dt']

        logger.info(f"Starting Simulation (T={duration}s, dt={dt}s)...")

        # 定义生成器供 Visualization 使用
        def sim_generator():
            t = 0.0
            while t < duration:
                self.step(t, dt)
                yield t, self.vehicles
                t += dt

                # 进度心跳
                if int(t) % 50 == 0 and abs(t - int(t)) < dt / 2:
                    logger.info(f"Simulating T={t:.0f}s...")

        # 启动带界面的仿真
        try:
            self.viz.start(sim_generator)
        except KeyboardInterrupt:
            logger.warning("Simulation stopped by user.")
        except Exception as e:
            logger.error(f"Runtime Error: {e}")
            raise e
        finally:
            self.save_data()

    def save_data(self):
        if not self.logs: return
        if not os.path.exists('data'): os.makedirs('data')

        df = pd.DataFrame(self.logs)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/SCI_Exp_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Experimental Data Saved: {filename} ({len(df)} rows)")


if __name__ == "__main__":
    sim = DecentralizedSimulation()
    sim.run()