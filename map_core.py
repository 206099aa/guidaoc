import networkx as nx
import numpy as np
import logging
from typing import Dict, List, Tuple

# 引入之前定义的边缘道岔智能体
from infrastructure import EdgeSwitchAgent

logger = logging.getLogger("MapCore")


class SpatialFieldGenerator:
    """
    [Environment Modeling]
    Generate spatially correlated environmental factors (e.g., Mud Depth).
    Uses 2D Gaussian Kernel convolution to simulate continuous field.
    """

    def __init__(self, width, height, seed=42):
        np.random.seed(seed)
        self.width = width
        self.height = height
        self.grid = self._generate_correlated_field()

    def _generate_correlated_field(self):
        # 1. White Noise
        noise = np.random.rand(self.width, self.height)

        # 2. Gaussian Filter (Smoothing for Correlation)
        # 模拟地理环境的连续性：泥泞通常是成片的
        x = np.arange(-2, 3)
        y = np.arange(-2, 3)
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
        kernel = kernel / np.sum(kernel)

        # Simple Convolution
        try:
            from scipy.signal import convolve2d
            field = convolve2d(noise, kernel, mode='same', boundary='symm')
        except ImportError:
            # Fallback if scipy is missing
            field = noise  # degraded mode

        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min())

        # Bias towards muddy (Paddy field characteristic)
        field = np.clip(field + 0.2, 0.0, 1.0)
        return field

    def get_value_at(self, x, y, max_dim_x, max_dim_y):
        # Map physical coordinate to grid index
        idx_x = int((x / max_dim_x) * (self.width - 1))
        idx_y = int((y / max_dim_y) * (self.height - 1))
        idx_x = np.clip(idx_x, 0, self.width - 1)
        idx_y = np.clip(idx_y, 0, self.height - 1)
        return self.grid[idx_x, idx_y]


class NodeObject:
    def __init__(self, node_id, pos, env_config, has_switch=False):
        self.id = node_id
        self.pos = pos
        self.agent = None

        # 如果是交叉口，部署边缘计算节点 (Switch Agent)
        if has_switch:
            # 每个道岔有独立的泥泞环境参数
            local_env = env_config.copy()
            # 这里的 mud_factor 会在 Map 构建时被覆盖为空间场的值
            self.agent = EdgeSwitchAgent(node_id, local_env)


class GridMap:
    """
    [Cyber-Physical Topology]
    Integrates Graph Topology with Environmental Fields and Edge Agents.
    """

    def __init__(self, config):
        self.cfg = config
        self.rows = config['topology']['rows']
        self.cols = config['topology']['cols']
        self.spacing = config['topology']['cell_spacing']

        self.graph = nx.DiGraph()
        self.nodes: Dict[str, NodeObject] = {}

        # 初始化空间场 (假设地图最大尺寸 2000m x 2000m)
        self.field_gen = SpatialFieldGenerator(20, 20)
        self.max_dim = max(self.rows, self.cols) * self.spacing

        self._build_topology()
        self._inject_heterogeneity()

    def _build_topology(self):
        # 1. 骨干网格 (Manhattan Grid)
        for r in range(self.rows):
            for c in range(self.cols):
                nid = f"N_{r}_{c}"
                x, y = c * self.spacing, r * self.spacing

                # 判断是否为路口 (拥有 >=3 连接的通常需要道岔)
                # 简化逻辑：所有骨干节点都部署智能道岔，以测试大规模协同
                self._add_node(nid, (x, y), has_switch=True)

        # 2. 连接边
        for r in range(self.rows):
            for c in range(self.cols):
                u = f"N_{r}_{c}"
                # Horizontal
                if c < self.cols - 1:
                    v = f"N_{r}_{c + 1}"
                    # 增加中间停靠点 (Loading Station)
                    stop_id = f"Stop_H_{r}_{c}"
                    mid_pos = ((self.nodes[u].pos[0] + (c + 1) * self.spacing) / 2, self.nodes[u].pos[1])
                    self._add_node(stop_id, mid_pos, has_switch=False)

                    self._add_edge(u, stop_id)
                    self._add_edge(stop_id, v)
                    # 双向轨道
                    self._add_edge(v, stop_id)
                    self._add_edge(stop_id, u)

                # Vertical
                if r < self.rows - 1:
                    v = f"N_{r + 1}_{c}"
                    self._add_edge(u, v)
                    self._add_edge(v, u)

        # 3. 车库 (Depots) - 从配置读取
        depots = self.cfg['topology'].get('depots', {})
        for name, coords in depots.items():
            self._add_node(name, tuple(coords), has_switch=False)
            # 寻找最近的骨干节点进行连接
            closest = min(self.nodes.keys(),
                          key=lambda n: np.linalg.norm(np.array(self.nodes[n].pos) - np.array(coords)))
            self._add_edge(name, closest)
            self._add_edge(closest, name)

    def _add_node(self, nid, pos, has_switch):
        # 获取该位置的局部环境参数
        local_mud = self.field_gen.get_value_at(pos[0], pos[1], self.max_dim, self.max_dim)

        # 注入环境配置
        node_env = self.cfg['environment'].copy()
        node_env['mud_factor'] = local_mud

        node_obj = NodeObject(nid, pos, node_env, has_switch)
        self.nodes[nid] = node_obj
        self.graph.add_node(nid, pos=pos)

    def _add_edge(self, u, v):
        # 计算边的物理属性
        p1 = np.array(self.nodes[u].pos)
        p2 = np.array(self.nodes[v].pos)
        dist = np.linalg.norm(p1 - p2)

        # [修复] 访问 agent.mud 而不是 agent.mud_factor
        m1 = self.nodes[u].agent.mud if self.nodes[u].agent else 0.5
        m2 = self.nodes[v].agent.mud if self.nodes[v].agent else 0.5
        avg_mud = (m1 + m2) / 2.0

        # 轨道不平顺度 (PSD 参数 A)
        roughness = 0.05 * (1.0 + 2.0 * avg_mud)

        self.graph.add_edge(u, v, weight=dist, length=dist, mud=avg_mud, roughness=roughness)

    def _inject_heterogeneity(self):
        logger.info("Spatial Heterogeneity Injected. Mud Factor Field Generated.")

    def update_infrastructure(self, dt, time):
        """主循环调用：更新所有边缘道岔的状态"""
        for node in self.nodes.values():
            if node.agent:
                node.agent.update(dt, time)