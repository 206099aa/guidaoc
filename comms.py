import numpy as np
import math
import logging
from interfaces import ILinkLayer

logger = logging.getLogger("StochasticChannel")


class AgriculturalCommChannel(ILinkLayer):
    """
    [Wireless Channel Model]
    Simulates V2I/V2N communication in Paddy Fields.
    Key characteristics: High humidity, vegetation obstruction, water reflection.
    """

    def __init__(self, tech: str, mud: float, emap: dict):
        self.tech = tech
        self.mud = mud
        self.energy_map = emap

        # 物理层参数
        if tech == "LoRa":
            self.freq = 433e6  # 433 MHz
            self.tx_power = 14  # dBm
            self.sensitivity = -130  # dBm
            self.rician_k = 10.0  # 强 LoS
        elif tech == "WiFi":
            self.freq = 2.4e9  # 2.4 GHz
            self.tx_power = 20  # dBm
            self.sensitivity = -85  # dBm
            self.rician_k = 3.0  # 较多散射
        else:
            raise ValueError(f"Unknown tech: {tech}")

        # 环境参数
        # 路径损耗指数 (Path Loss Exponent)
        # 泥泞/潮湿环境吸波更强，指数更大
        self.n_path = 2.5 + 0.5 * mud

        # 阴影衰落标准差 (dB)
        self.sigma_shadow = 4.0

    def _rician_fading(self, K_dB):
        """
        生成莱斯衰落样本 (复高斯分布)
        """
        K = 10 ** (K_dB / 10.0)
        mu = math.sqrt(K / (2 * (K + 1)))
        sigma = math.sqrt(1 / (2 * (K + 1)))

        # h = (x + j*y) + LoS_Component
        h_los = math.sqrt(K / (K + 1))
        h_nlos = complex(np.random.normal(0, sigma), np.random.normal(0, sigma))

        envelope = abs(h_los + h_nlos)
        # Normalize power to 1
        return 20 * math.log10(envelope + 1e-9)

    def transmit(self, dist: float, size: int, v: float) -> tuple:
        """
        执行一次传输仿真
        Returns: (success, rssi, energy_cost, latency, debug_info)
        """
        dist = max(dist, 1.0)

        # 1. 路径损耗 (Log-distance Path Loss)
        # PL(d) = PL(d0) + 10n * log10(d/d0)
        lambda_wave = 3e8 / self.freq
        pl_d0 = 20 * math.log10(4 * math.pi * 1.0 / lambda_wave)  # Friis at 1m
        pl = -pl_d0 + 10 * self.n_path * math.log10(dist)

        # 2. 阴影衰落 (Log-normal Shadowing) - 慢衰落
        shadowing = np.random.normal(0, self.sigma_shadow)

        # 3. 多径衰落 (Rician Fading) - 快衰落
        # 车辆速度越快，多普勒频移越大，信道变化越快 (此处简化为每帧独立采样)
        fading = self._rician_fading(self.rician_k)

        # 4. 天线增益损耗 (Antenna Mismatch due to Mud)
        # 泥浆覆盖天线会导致 SWR 升高
        loss_ant = 5.0 * self.mud

        # 计算接收信号强度 (RSSI)
        rssi = self.tx_power - pl + shadowing + fading - loss_ant

        # 5. 信噪比与误包率 (SNR -> BER -> PER)
        noise_floor = -174 + 10 * math.log10(125e3 if self.tech == "LoRa" else 20e6) + 10  # Noise Figure
        snr = rssi - noise_floor

        # 简化的 BER 模型 (BPSK/CSS)
        # LoRa 在 SNR < -10 仍可工作, WiFi 需要 SNR > 5
        limit = -20.0 if self.tech == "LoRa" else 0.0
        success_prob = 1.0 / (1.0 + np.exp(-(snr - limit)))  # Sigmoid transition

        is_success = np.random.random() < success_prob

        # 能耗计算 (Tx Duration * Power)
        # Time on Air (ToA)
        datarate = 5000 if self.tech == "LoRa" else 6e6  # bps
        latency = (size * 8) / datarate

        # Power (Watts) = 10^(dBm/10) / 1000
        p_watt = (10 ** (self.tx_power / 10)) / 1000
        energy = p_watt * latency

        return is_success, rssi, energy, latency, {'snr': snr}

    def get_diagnostics(self):
        return {'tech': self.tech}