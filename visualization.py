import matplotlib

# 强制使用 TkAgg 后端以支持实时弹窗动画
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from collections import deque


class SimVisualizer:
    """
    [Visualization Layer] SCI-Grade Digital Twin Interface.
    Integrates Spatio-Temporal Fields, Micro-Physics, and Real-time Telemetry Matrix.
    """

    def __init__(self, grid_map, vehicles, config):
        self.grid = grid_map;
        self.vehicles = vehicles;
        self.cfg = config

        self.fig = plt.figure(figsize=(19, 12), facecolor='#f0f0f0')
        exp_id = config['meta'].get('experiment_id', 'N/A')
        self.fig.suptitle(f"SCI Digital Twin: Physics & Telemetry (Exp: {exp_id})", fontsize=16, fontweight='bold')

        # [SCI Layout] 3行布局：
        # Row 0: Map (左2/3) + Force/Mu Plot (右1/3)
        # Row 1: Phase Plane (左) + Sinkage (中) + RSSI (右)
        # Row 2: Telemetry Table (全宽)
        gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 0.6])

        # 1. 宏观地图
        self.ax_map = self.fig.add_subplot(gs[0, :2])
        self.ax_map.set_title("Spatio-Temporal Field & Agent Tracking")
        self.ax_map.set_aspect('equal')

        # 2. 纵向动力学 (双轴：力 & 摩擦)
        self.ax_phys = self.fig.add_subplot(gs[0, 2])
        # [Fix 1] 使用 raw string 修复警告
        self.ax_phys.set_title(r"Coupler Force (L) vs Friction $\mu$ (R)")
        self.ax_mu = self.ax_phys.twinx()

        # 3. 三个子图
        self.ax_phase = self.fig.add_subplot(gs[1, 0])
        self.ax_phase.set_title("Stability: Phase Plane")
        self.ax_phase.set_xlabel("e");
        self.ax_phase.set_ylabel("de/dt")
        self.ax_phase.grid(True)

        self.ax_sink = self.fig.add_subplot(gs[1, 1])
        self.ax_sink.set_title("Terramechanics: Sinkage")
        self.ax_sink.set_ylim(0, 0.4);
        self.ax_sink.grid(True)

        self.ax_rssi = self.fig.add_subplot(gs[1, 2])
        self.ax_rssi.set_title("Comm: RSSI (dBm)")
        self.ax_rssi.set_ylim(-130, -40);
        self.ax_rssi.grid(True)

        # 4. 遥测矩阵 (底部全宽)
        self.ax_table = self.fig.add_subplot(gs[2, :])
        self.ax_table.axis('off')
        self.ax_table.set_title("Real-time Vehicle Telemetry Matrix (SCI Q1 Metrics)", fontweight='bold', y=1.0)

        # 缓存
        self.hist_len = 200
        self.t_hist = deque(maxlen=self.hist_len)
        self.d_store = {v.id: {
            'force': deque(maxlen=self.hist_len), 'mu': deque(maxlen=self.hist_len),
            'err': deque(maxlen=self.hist_len), 'err_d': deque(maxlen=self.hist_len),
            'sink': deque(maxlen=self.hist_len), 'rssi': deque(maxlen=self.hist_len)
        } for v in vehicles}

        self.dyn_objs = []
        self._init_static()
        self._init_table()

    def _init_static(self):
        ax = self.ax_map
        for u, v, d in self.grid.graph.edges(data=True):
            p1, p2 = self.grid.nodes[u].pos, self.grid.nodes[v].pos
            c = plt.cm.YlOrBr(d.get('mud', 0.5))
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c=c, lw=3, zorder=1)
        for nid, n in self.grid.nodes.items():
            if n.agent:
                ax.scatter(*n.pos, marker='D', s=80, c='gray', zorder=2)
            elif "Start" in nid:
                ax.text(n.pos[0], n.pos[1], nid, fontsize=8)

    def _init_table(self):
        # [SCI Columns Definition]
        self.cols = [
            "ID", "Mass (kg)", "Len (m)",  # 静态
            "μ_eff (1)", "Mud Fac",  # 环境交互
            "v_inst (km/h)", "v_avg (km/h)",  # 速度
            "P_inst (kW)", "P_avg (kW)",  # 功率
            "E_tot (kJ)", "SEC (J/kg·m)"  # 能效
        ]
        cell_text = [["-" for _ in self.cols] for _ in self.vehicles]
        self.table = self.ax_table.table(
            cellText=cell_text, colLabels=self.cols, loc='center', cellLoc='center',
            bbox=[0.0, 0.0, 1.0, 0.9], colColours=["#e0e0e0"] * len(self.cols)
        )
        self.table.auto_set_font_size(False);
        self.table.set_fontsize(9)

    def update(self, data):
        t, vehicles = data
        self.t_hist.append(t)

        for o in self.dyn_objs:
            try:
                o.remove()
            except:
                pass
        self.dyn_objs.clear()

        tab_vals = []

        for v in vehicles:
            # --- 1. Map Render ---
            c = 'lime' if v.state.name == "TRACTION_CONTROL" else 'gold'
            r = patches.Rectangle((v.pos_2d[0] - 10, v.pos_2d[1] - 5), 20, 10, fc=c, ec='k', zorder=5)
            self.ax_map.add_patch(r);
            self.dyn_objs.append(r)
            txt = self.ax_map.text(v.pos_2d[0], v.pos_2d[1] + 12, v.id, ha='center', fontsize=7)
            self.dyn_objs.append(txt)

            # --- 2. Data Processing ---
            tm = getattr(v, 'last_telemetry', {})
            st = self.d_store[v.id]

            # Extract
            mu = tm.get('mu', 0);
            f = tm.get('force', 0) / 1000
            vel = tm.get('vel', 0);
            p_inst = tm.get('p_inst', 0)
            e_tot = tm.get('energy_total', 0);
            dist = tm.get('dist_accum', 0)
            t_act = tm.get('time_active', 0.1)
            mud = tm.get('mud', 0.5)
            mass = tm.get('mass', 0)

            # Update Hist
            st['force'].append(f);
            st['mu'].append(mu)
            st['sink'].append(0.05 * mud / (abs(vel) + 0.5))
            st['rssi'].append(tm.get('rssi', -90))
            err = 15.0 - vel;
            st['err'].append(err)
            st['err_d'].append(err - (st['err'][-2] if len(st['err']) > 1 else err))

            # Calc SCI Metrics
            v_kmh = vel * 3.6
            v_avg = (dist / t_act * 3.6) if t_act > 1 else 0.0
            p_kw = p_inst / 1000.0
            p_avg = (e_tot / t_act / 1000.0) if t_act > 1 else 0.0
            sec = (e_tot / (mass * dist)) if dist > 10 else 0.0

            # [SCI High Precision] 提高显示精度到 4 位小数，观察微动
            tab_vals.append([
                str(v.id), f"{mass:.0f}", f"{tm.get('length', 0):.1f}",
                f"{mu:.3f}", f"{mud:.2f}",
                f"{v_kmh:.4f}", f"{v_avg:.2f}",
                f"{p_kw:.1f}", f"{p_avg:.1f}",
                f"{e_tot / 1000:.0f}", f"{sec:.4f}"
            ])

        # --- 3. Update Table ---
        for r, row in enumerate(tab_vals):
            for c, val in enumerate(row):
                self.table[r + 1, c].get_text().set_text(val)

        # --- 4. Update Plots (Only 1st vehicle to reduce clutter) ---
        if vehicles:
            vid = vehicles[0].id;
            s = self.d_store[vid]
            if len(self.t_hist) == len(s['force']):
                self.ax_phys.clear();
                self.ax_mu.clear()
                # [Fix 1] 使用 raw string 修复警告
                self.ax_phys.set_title(r"Force (L) vs $\mu$ (R)");
                self.ax_phys.grid(True, alpha=0.3)
                self.ax_phys.plot(self.t_hist, s['force'], 'b-', alpha=0.6)
                self.ax_mu.plot(self.t_hist, s['mu'], 'r--', alpha=0.6)

                self.ax_phase.clear();
                self.ax_phase.set_title("Phase Plane");
                self.ax_phase.grid(True)
                self.ax_phase.plot(s['err'], s['err_d'], 'g-', alpha=0.5)

                self.ax_sink.clear();
                self.ax_sink.set_title("Sinkage");
                self.ax_sink.grid(True)
                self.ax_sink.set_ylim(0, 0.4);
                self.ax_sink.plot(self.t_hist, s['sink'], 'brown')

                self.ax_rssi.clear();
                self.ax_rssi.set_title("RSSI");
                self.ax_rssi.grid(True)
                self.ax_rssi.set_ylim(-130, -40);
                self.ax_rssi.plot(self.t_hist, s['rssi'], 'purple')

    def start(self, gen):
        # [Fix 2] 添加 cache_frame_data=False 修复内存警告
        ani = animation.FuncAnimation(
            self.fig, self.update, frames=gen,
            interval=20, blit=False, repeat=False,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()