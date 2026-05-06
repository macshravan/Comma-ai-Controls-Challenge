"""
controllers/koopman_mpc.py
--------------------------
Koopman-MPC Lateral Controller.

Architecture:
    - Uses Koopman operator to identify a velocity-dependent linear gain
      mapping from error state to steer command
    - Wraps a PI controller around this gain for stability
    - Online Koopman refitting adapts the gain to the current segment

This is a data-driven gain-scheduled controller:
    steer = K_koopman(v) * error + integral_term
"""

import numpy as np
import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from koopman_operator import KoopmanOperator, lift, N_OBS

State      = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])


class Controller:
    """
    Koopman gain-scheduled PI lateral controller.

    The Koopman operator identifies how the system responds to steer
    commands as a function of the lifted state. From the B matrix we
    extract a velocity-dependent gain that drives the PI controller.
    """

    def __init__(self,
                 steer_limit: float = 2.0,
                 koopman_path: str  = None,
                 Kp: float          = 0.4,
                 Ki: float          = 0.02,
                 Kd: float          = 0.1,
                 dt: float          = 0.1):

        self.steer_limit = steer_limit
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        # Load Koopman operator
        self.ko = KoopmanOperator()
        if koopman_path is None:
            koopman_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'koopman_trained.npz')
        if os.path.exists(koopman_path):
            self.ko.load(koopman_path)
            # Simulator negates steerCommand — flip B sign
            self.ko.B = -self.ko.B
            self._koopman_gain = self._extract_koopman_gain()
            pass  # gain identified silently
        else:
            self._koopman_gain = 1.0

        # PI state
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_steer = 0.0
        self._steps      = 0

        # Online buffer
        self._state_buf  = []
        self._action_buf = []
        self._refit_every = 80

    def _extract_koopman_gain(self):
        """
        Extract scalar gain from Koopman B matrix.
        B[0] tells us how much steer affects latAccel in lifted space.
        We use this to scale our base PI gains.
        """
        b_lataccel = float(self.ko.B[0, 0])
        if abs(b_lataccel) < 1e-6:
            return 1.0
        # Gain = 1 / sensitivity (how much steer we need per unit latAccel error)
        gain = abs(1.0 / b_lataccel)
        return np.clip(gain, 0.1, 20.0)

    def _velocity_scale(self, v_ego):
        """Scale gains with velocity."""
        v = max(v_ego, 1.0)
        return np.clip(v / 15.0, 0.5, 2.0)

    def _build_state(self, current_lataccel, state):
        return np.array([
            current_lataccel,
            state.v_ego,
            state.a_ego,
            state.roll_lataccel,
            self._prev_steer
        ])

    def _refit_koopman(self):
        """Online refit and update gain."""
        if len(self._state_buf) < 30:
            return
        try:
            X = np.stack(self._state_buf, axis=0)
            U = np.array(self._action_buf[:-1]).reshape(-1, 1)
            if X.shape[0] - 1 == U.shape[0]:
                self.ko.fit(X, U)
                new_gain = self._extract_koopman_gain()
                # Smooth update
                self._koopman_gain = 0.7 * self._koopman_gain + 0.3 * new_gain
        except Exception:
            pass

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """Main controller update."""
        self._steps += 1

        x = self._build_state(current_lataccel, state)
        self._state_buf.append(x.copy())
        if len(self._state_buf) > 200:
            self._state_buf.pop(0)
        if len(self._action_buf) > 200:
            self._action_buf.pop(0)

        # Periodic online refit
        if self._steps % self._refit_every == 0:
            self._refit_koopman()

        # Compute error
        error = target_lataccel - current_lataccel

        # Velocity scaling
        v_scale = self._velocity_scale(state.v_ego)

        # Koopman-scaled PI controller
        kp = self.Kp * v_scale
        ki = self.Ki * v_scale
        kd = self.Kd

        # Integral with anti-windup
        self._integral += error * self.dt
        self._integral  = np.clip(self._integral, -2.0, 2.0)

        # Derivative
        deriv = (error - self._prev_error) / self.dt
        self._prev_error = error

        # PID output scaled by Koopman gain
        # Koopman gain tells us how much steer is needed per unit latAccel
        raw_output = (kp * error + ki * self._integral + kd * deriv)

        # Apply Koopman gain scaling — normalize by sensitivity
        u_opt = raw_output * min(self._koopman_gain / 10.0, 2.0)

        u_opt = float(np.clip(u_opt, -self.steer_limit, self.steer_limit))

        self._action_buf.append(u_opt)
        self._prev_steer = u_opt

        return u_opt
