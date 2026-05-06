from . import BaseController
import numpy as np

# AR model parameters identified from ONNX physics (identify_system.py)
# la[t+1] = AR_A*la[t] + AR_B*steer[t] + AR_C*roll[t] + AR_D
AR_A = 0.9577
AR_B = 0.0605
AR_C = 0.0404
AR_D = -0.007828


# Empirical steady-state feedforward gain: steer_ss ≈ FF_GAIN * target_lataccel
# Derived from 100 real data files (steps 0-100, openpilot tracking well)
FF_GAIN = 0.31498135945886213

class Controller(BaseController):
  """
  Feedforward + PID controller for lateral acceleration tracking.

  Key improvements over naive PID:
  1. Feedforward: steer_ff = FF_GAIN * effective_target removes steady-state lag
  2. Preview: blends current target with mean of next N future targets (70/30)
     so the controller anticipates upcoming turns before they happen
  3. Correct future_plan access via future_plan.lataccel (namedtuple field)
  4. Steer range extended to [-2, 2] (was incorrectly capped at [-1, 1])
  5. PID only handles residual error after feedforward
  """

  def __init__(self):
    # PID gains (higher P is OK because FF handles steady-state,
    # so PID only corrects small residual errors)
    self.p = 0.9704110065903847
    self.i = 0.07595479827299831
    self.d = 0.5

    # Feedforward settings
    self.ff_gain = FF_GAIN
    self.future_weight = 0.9492317622605102   # blend: 30% current target, 70% future avg
    self.n_future = 10          # how many future steps to average

    # Controller state
    self.error_integral = 0.0
    self.prev_error = 0.0
    self.prev_output = 0.0

    # Rate limit: max steer change per step (keeps jerk low)
    self.max_delta = 0.2

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # --- PREVIEW: blend current target with near-future average ---
    # future_plan.lataccel is a list of upcoming target lataccels
    # (FIX: was future_plan[:3] which sliced namedtuple *fields*, not timesteps)
    if future_plan is not None and len(future_plan.lataccel) >= self.n_future:
      future_avg = np.mean(future_plan.lataccel[:self.n_future])
      effective_target = (1.0 - self.future_weight) * target_lataccel + \
                          self.future_weight * future_avg
    else:
      effective_target = target_lataccel

    # --- FEEDFORWARD: steady-state steer for effective target ---
    steer_ff = self.ff_gain * effective_target

    # --- PID: correct residual tracking error ---
    error = target_lataccel - current_lataccel

    self.error_integral += error
    self.error_integral = np.clip(self.error_integral, -5, 5)

    error_diff = error - self.prev_error
    self.prev_error = error

    pid_output = (
        self.p * error +
        self.i * self.error_integral +
        self.d * error_diff
    )

    # --- COMBINE feedforward + PID ---
    raw_output = steer_ff + pid_output

    # --- SMOOTH + RATE LIMIT (preserves jerk characteristics) ---
    output = 0.85 * self.prev_output + 0.15 * raw_output
    output = np.clip(output,
                     self.prev_output - self.max_delta,
                     self.prev_output + self.max_delta)

    self.prev_output = output

    # FIX: steer range is [-2, 2], not [-1, 1]
    return np.clip(output, -2, 2)
