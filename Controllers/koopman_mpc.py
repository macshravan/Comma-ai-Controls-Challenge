from . import BaseController
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
A = np.load("A.npy")
B = np.load("B.npy")
C = np.load("C.npy")

x_sc = joblib.load("x_scaler.pkl")
u_sc = joblib.load("u_scaler.pkl")
phi  = joblib.load("phi.pkl")
cfg  = joblib.load("koopman_cfg.pkl")

DELAY = cfg["DELAY"]
ALPHA = cfg["BLEND_ALPHA"]

FF_GAIN = 0.31498135945886213

# =========================
# MPC PARAMS
# =========================
H = 8
N_CANDIDATES = 30        # modest increase from 15
LAMBDA_JERK = 12.0       # reduced from 35, but NOT as low as 4
LAMBDA_U = 0.8           # reduced from 1.5


class Controller(BaseController):

  def __init__(self):
    self.p = 0.9          # modest increase from 0.7
    self.i = 0.03
    self.d = 0.2

    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_output = 0.0

    self.max_delta = 0.25  # very close to original 0.2

    self.state_buffer = []

  def extract_state(self, state, current_lataccel):
    return np.array([
      state.v_ego,
      state.a_ego,
      state.roll_lataccel,
      current_lataccel
    ], dtype=np.float64)

  def get_z(self):
    if len(self.state_buffer) < DELAY:
      return None

    x_hist = np.array(self.state_buffer[-DELAY:])
    x_scaled = x_sc.transform(x_hist)
    x_scaled = x_scaled[::-1].reshape(1, -1)

    z = phi.transform(x_scaled).flatten()
    return z

  def solve_mpc(self, z, targets, u_prev):
    # back to stable spread logic
    spread = 0.6 + 0.8 * abs(u_prev)

    u_candidates = np.concatenate([
      np.linspace(u_prev - 0.2, u_prev + 0.2, N_CANDIDATES // 2),
      np.linspace(u_prev - spread, u_prev + spread, N_CANDIDATES // 2)
    ])
    u_candidates = np.clip(u_candidates, -2, 2)

    best_cost = 1e9
    best_u = u_prev

    for u in u_candidates:
      z_r = z.copy()
      # fix: use proper Koopman output for initial y
      y_r = C[0] @ z_r
      cost = 0.0
      u_last = u_prev

      for t in range(H):
        target_t = targets[t] if t < len(targets) else targets[-1]

        u_scaled = u_sc.transform([[u]])[0, 0]
        z_r = A @ z_r + B.flatten() * u_scaled

        y_abs = C[0] @ z_r
        dy    = C[1] @ z_r
        y_r   = ALPHA * (y_r + dy) + (1 - ALPHA) * y_abs

        cost += (y_r - target_t) ** 2
        cost += LAMBDA_JERK * (u - u_last) ** 2
        cost += LAMBDA_U * (u ** 2)

        u_last = u

      if cost < best_cost:
        best_cost = cost
        best_u = u

    return best_u

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    x = self.extract_state(state, current_lataccel)
    self.state_buffer.append(x)
    if len(self.state_buffer) > DELAY:
      self.state_buffer.pop(0)

    z = self.get_z()

    if future_plan is not None and len(future_plan.lataccel) > 0:
      targets = list(future_plan.lataccel[:H])
      while len(targets) < H:
        targets.append(targets[-1])
    else:
      targets = [target_lataccel] * H

    if z is None:
      steer = FF_GAIN * target_lataccel
    else:
      steer = self.solve_mpc(z, targets, self.prev_output)

    # =========================
    # PID — modest role
    # =========================
    error = target_lataccel - current_lataccel

    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -5, 5)

    error_diff = error - self.prev_error
    self.prev_error = error

    pid = (
      self.p * error +
      self.i * self.error_integral +
      self.d * error_diff
    )

    raw = steer + 0.3 * pid   # was 0.25, slight bump

    # KEY FIX: smoothing back to stable range, but less extreme than 0.85
    output = 0.75 * self.prev_output + 0.25 * raw

    output = np.clip(
      output,
      self.prev_output - self.max_delta,
      self.prev_output + self.max_delta
    )

    self.prev_output = output

    return np.clip(output, -2, 2)
