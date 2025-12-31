import networkx as nx
import logging
import math

logger = logging.getLogger("SmartRouter")


class IntelligentRouter:
    """
    [Path Planning] Energy-Risk Weighted A* Algorithm.
    Objective: Minimize J = alpha * Energy + beta * Risk + gamma * Time
    """

    def __init__(self, grid_map):
        self.grid = grid_map
        self.graph = grid_map.graph

    def _cost_function(self, u, v, edge_attrs):
        """
        [Novelty] Dynamic Weighting based on Physical State.
        """
        # 1. 基础物理代价: 距离
        dist = edge_attrs.get('weight', 100.0)

        # 2. 环境阻力代价: 泥泞度 (Energy Proxy)
        # 泥越厚，能耗越高，权重越大
        mud = edge_attrs.get('mud', 0.5)
        friction_cost = 1.0 + 5.0 * mud  # 泥路代价翻倍

        # 3. 基础设施风险代价
        node_v = self.grid.nodes[v]
        risk_penalty = 0.0

        if node_v.agent:  # 如果是智能道岔
            # 读取边缘健康度
            health = node_v.agent.health_index  # 1.0 = Good, 0.0 = Bad

            # 如果道岔亚健康，施加惩罚以绕行
            if health < 0.6:
                risk_penalty = 500.0 * (1.0 - health)

            # 如果道岔已故障，视为断路
            if node_v.agent.state.name == 'STALLED':
                return float('inf')

        # 综合权重
        return dist * friction_cost + risk_penalty

    def _heuristic(self, u, v):
        """Manhattan Distance Heuristic"""
        p1 = self.grid.nodes[u].pos
        p2 = self.grid.nodes[v].pos
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def get_dynamic_path(self, start_id, end_id, vehicle_id, scheduler_ref=None):
        """
        Compute optimal path considering current field conditions.
        """
        try:
            path = nx.astar_path(
                self.graph,
                start_id,
                end_id,
                heuristic=self._heuristic,
                weight=self._cost_function
            )
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path found for {vehicle_id} from {start_id} to {end_id}")
            return []
        except Exception as e:
            logger.error(f"Routing Error: {e}")
            return []