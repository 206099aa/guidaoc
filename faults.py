import numpy as np
import logging
from scipy.stats import gamma

logger = logging.getLogger("ReliabilityPhysics")


class DegradationModel:
    """
    [Reliability Model] Gamma Process for Monotonic Degradation.
    Models the gradual wear of mechanical components (e.g., switch gears, axles).
    Reference: Van Noortwijk (2009) "A survey of the application of gamma processes in maintenance".
    """

    def __init__(self, shape_param, scale_param, failure_threshold=10.0):
        self.k = shape_param  # Shape (wear rate)
        self.theta = scale_param  # Scale (variance)
        self.threshold = failure_threshold
        self.cumulative_wear = 0.0
        self.is_failed = False

    def step(self, dt, stress_factor):
        """
        :param stress_factor: 环境应力 (如泥泞度 * 负载)
        """
        if self.is_failed: return True

        # 磨损增量服从 Gamma 分布: Delta ~ Gamma(k * dt * stress, theta)
        # 注意：Gamma 分布的参数需根据 dt 缩放
        shape = self.k * dt * stress_factor
        if shape > 0:
            wear_increment = np.random.gamma(shape, self.theta)
            self.cumulative_wear += wear_increment

        if self.cumulative_wear >= self.threshold:
            self.is_failed = True

        return self.is_failed

    def get_health_index(self):
        # 归一化健康度 1.0 -> 0.0
        return max(0.0, 1.0 - self.cumulative_wear / self.threshold)


class ReliabilityEngine:
    """
    [PHM Engine] Prognostics and Health Management.
    Injects realistic faults into sensors and actuators based on environmental stress.
    """

    def __init__(self, env_config, vehicle_config):
        self.mud = env_config.get('mud_factor', 0.5)

        # 1. 执行器退化模型 (Actuator Wear)
        # 泥浆越厚，磨损越快
        self.actuator_wear = DegradationModel(shape_param=0.01, scale_param=0.05, failure_threshold=100.0)

        # 2. 传感器状态
        self.sensor_bias = 0.0
        self.sensor_noise_std = 0.05  # 基础噪声

    def update_health(self, dt, current_load_norm):
        """
        更新系统健康状态
        :param current_load_norm: 归一化负载 (Current / Max_Current)
        """
        # 应力因子 = 基础负荷 + 泥浆环境惩罚
        stress = current_load_norm * (1.0 + 2.0 * self.mud)

        # 1. 机械磨损步进
        is_hard_fail = self.actuator_wear.step(dt, stress)

        # 2. 传感器渐变故障 (Drift)
        # 泥浆逐渐覆盖传感器，导致 Bias 线性增加
        self.sensor_bias += 0.0001 * self.mud * dt

        # 返回故障概率 (用于软故障判定)
        # 随着磨损增加，随机故障概率指数上升
        hi = self.actuator_wear.get_health_index()
        random_fail_prob = 1.0 * np.exp(-10.0 * hi) * dt

        return random_fail_prob

    def check_hard_failure(self, fail_prob):
        """判断是否发生硬故障"""
        if self.actuator_wear.is_failed:
            return True, "WEAR_OUT"

        if np.random.random() < fail_prob:
            return True, "RANDOM_FAILURE"

        return False, "OK"

    def apply_sensor_degradation(self, true_value, sensor_type='velocity'):
        """
        [Sensor Model] Add Bias, Noise, and Outliers.
        y = x + bias(t) + N(0, sigma(mud))
        """
        # 泥泞导致信噪比下降 (噪声方差变大)
        current_std = self.sensor_noise_std * (1.0 + 3.0 * self.mud)

        noise = np.random.normal(0, current_std)
        drift = self.sensor_bias

        # 离群点 (Outlier): 模拟泥块甩到传感器上导致的瞬间跳变
        if np.random.random() < 0.001 * self.mud:
            noise *= 10.0

        return true_value + drift + noise