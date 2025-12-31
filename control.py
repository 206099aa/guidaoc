import numpy as np


class KalmanFilter:
    """State Estimation for Noisy Sensors"""

    def __init__(self, dt):
        self.dt = dt
        self.A = np.array([[1, dt, 0.5 * dt ** 2], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([[0, 1, 0]])  # Measure velocity only
        self.x_hat = np.zeros((3, 1))
        self.P = np.eye(3)
        self.Q = np.eye(3) * 0.1  # Process noise
        self.R = np.array([[0.5]])  # Measurement noise

    def update(self, z):
        # Predict
        self.x_hat = self.A @ self.x_hat
        self.P = self.A @ self.P @ self.A.T + self.Q
        # Correct
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x_hat = self.x_hat + K @ (z - self.H @ self.x_hat)
        self.P = (np.eye(3) - K @ self.H) @ self.P
        return self.x_hat[1, 0], self.x_hat[2, 0]


class FuzzyAdaptivePID:
    """Mamdani Fuzzy Logic for Gain Scheduling"""

    def __init__(self, base_kp, base_ki, base_kd, max_out):
        self.kp, self.ki, self.kd = base_kp, base_ki, base_kd
        self.max_out = max_out
        self.integral = 0.0
        self.kf = KalmanFilter(0.1)

    def compute(self, target, current_noisy, dt):
        est_vel, est_acc = self.kf.update(current_noisy)
        error = target - est_vel

        # Fuzzy Logic Simulation (Simplified Rule Base)
        # Error Large -> Increase Kp
        # Error Small -> Increase Ki
        adapt_p = 1.0 + abs(error) * 0.5
        adapt_i = 1.0 / (1.0 + abs(error))

        p = self.kp * adapt_p * error
        self.integral += error * dt
        i = self.ki * adapt_i * self.integral
        d = self.kd * (-est_acc)

        out = np.clip(p + i + d, -self.max_out, self.max_out)
        return out, {'est_vel': est_vel, 'kp_adapt': adapt_p}


class SlidingModeController:
    """Robust Control Baseline: u = u_eq + K*sgn(s)"""

    def __init__(self, mass, max_force, lambda_param, k_gain):
        self.mass = mass
        self.max_force = max_force
        self.lam = lambda_param
        self.K = k_gain
        self.phi = 0.5  # Boundary layer

    def compute(self, target, current, dt):
        error = target - current
        s = error  # Sliding surface (simplified)

        # Equivalent control (Model inversion)
        u_eq = 200.0 * current + 500.0  # Compensation for resistance

        # Switching control (Robustness)
        u_sw = self.K * np.clip(s / self.phi, -1.0, 1.0)

        out = np.clip(u_eq + u_sw, -self.max_force, self.max_force)
        return out, {'s': s}