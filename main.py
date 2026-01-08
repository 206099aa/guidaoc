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
        self.infra_agents = {
            nid: node.agent
            for nid, node in self.map.nodes.items()
            if node.agent is not None
        }
        logger.info(f"Deployed {len(self.infra_agents)} Edge Switch Agents.")

        # 2. 初始化车辆智能体 (Vehicle Agents)
        logger.info("Deploying Autonomous Vehicles (4 Units)...")
        self.vehicles = []

        # 定义车辆编队配置 (4车: 每个起点各2辆，混编)
        fleet_config = [
            # --- Group 1: From Start_1 ---
            {
                "id": "HV_Hauler_01",
                "type": "Heavy_Hauler",
                "start": "Start_1"
            },
            {
                "id": "FS_Scout_01",
                "type": "Fast_Scout",
                "start": "Start_1"
            },
            # --- Group 2: From Start_2 ---
            {
                "id": "HV_Hauler_02",
                "type": "Heavy_Hauler",
                "start": "Start_2"
            },
            {
                "id": "FS_Scout_02",
                "type": "Fast_Scout",
                "start": "Start_2"
            }
        ]

        # 批量实例化
        for v_conf in fleet_config:
            v_agent = VehicleAgent(
                agent_id=v_conf["id"],
                vehicle_type_cfg=self.cfg['vehicle_types'][v_conf["type"]],
                env_config=self.env_cfg,
                start_node=v_conf["start"],
                map_graph=self.map,
                infra_agents=self.infra_agents
            )
            self.vehicles.append(v_agent)
            logger.info(f" -> Deployed {v_conf['id']} at {v_conf['start']}")

        # [SCI Depth] 注入 V2V 引用 (全连接拓扑，用于模拟硬件广播)
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
            step_count = 0

            # [视觉优化] 渲染倍速设置 (100倍物理帧跳过)
            # 相当于 50 倍速播放，避免等待
            RENDER_SKIP = 100

            while t < duration:
                self.step(t, dt)

                step_count += 1

                # 只有当计数器整除 RENDER_SKIP 时，才向可视化界面发送数据
                if step_count % RENDER_SKIP == 0:
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