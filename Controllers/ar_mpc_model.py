"""
Koopman + Incremental Error-Tracking MPC  (V2 — corrected)
===========================================================

Architecture
------------
steer[t] = steer_ff[t]  +  Δu*[t]

  steer_ff[t] = FF_GAIN * eff_target[t]          ← feedforward base (handles SS)
  Δu*[t]      = MPC correction for tracking error ← disturbance rejection only

Why incremental / why NOT tracking future targets with MPC
-----------------------------------------------------------
  The raw AR model has b≈0.06 (short-horizon fit), but TinyPhysics needs
  b_eff≈0.14 at steady state (implied by FF_GAIN=0.30).  MPC with the
  wrong b massively over-steers → oscillation → score ~13,000.

  Key insight: when the full free-term includes future target changes,
  k_free[0] ≈ −2.3 so a 1 m/s² target step triggers Δu≈2.3 which is
  instantly capped by rate-limit → MPC optimal solution never applied
  → pure oscillation.

  Fix: let the FF's 70/30 future-blend handle target anticipation.
  The MPC's free_vec only carries roll disturbance + constant bias:
      free[k] = AR_CROLL * roll[k] + AR_D
  These are always tiny (< 0.005 per step), so k_free × free_vec ≈ 0
  and the controller degenerates to a clean proportional-derivative
  error feedback — stable, no clipping, beats my_controller.

Error dynamics (AR-1 in error space)
-------------------------------------
  e[t]   = la[t] − target[t]
  Δu[t]  = steer[t] − steer_ff[t]

  e[t+1] ≈ AR_A·e[t] + b_correct·Δu[t] + free[t]

  free[t] = AR_CROLL·roll[t] + AR_D           ← disturbances only

  b_correct = (1 − AR_A) / FF_GAIN  →  free ≈ 0 at steady state ✓

MPC formulation
---------------
  Horizon H = 20.  State = scalar e[k], control = scalar Δu[k].

  Predict E = Sx·e₀ + Su·ΔU + Sd·free_vec
  Cost: J = Q·‖E‖² + R_Δ·‖D·ΔU − Δu_prev·e₁‖² + R_e·‖ΔU‖²

  Closed-form: Δu*[0] = k_e0·e₀ + k_free·f + k_uprev·Δu_prev
  All scalars/vectors precomputed once at class definition.

Output
------
  steer = clip(steer_ff + smooth(Δu*[0]), −2, 2)
"""

from . import BaseController
import numpy as np

# ── AR(1) model parameters ────────────────────────────────────────────────────
# From identify_system.py (fitted on 50 SYNTHETIC files via PID rollouts)
AR_A     = 0.9577      # la autoregressive coefficient
AR_CROLL = 0.0404      # roll_lataccel → la gain
AR_D     = -0.007828   # constant bias (la/step)

# Feedforward gain: steer=0.30 achieves la=1.0 at TinyPhysics steady state
FF_GAIN  = 0.30

# b_correct calibrated so that (AR_A + b_correct*FF_GAIN) = 1.0
# → free-term = 0 at constant-target, zero-roll steady state
AR_B_CORRECT = (1.0 - AR_A) / FF_GAIN    # ≈ 0.1410

# ── MPC hyper-parameters ─────────────────────────────────────────────────────
H          = 20      # prediction horizon (steps)
Q_TRACK    = 50.0    # error tracking weight
R_DELTA    = 7.0   # steer-increment rate penalty  ← best found by sweep
R_EFFORT   = 0.02    # small effort regularisation
STEER_MAX  = 2.0

# ── FF future-blend (mirrors my_controller.py) ────────────────────────────────
FF_FUTURE_WEIGHT = 0.70    # fraction on future average vs current target
FF_N_FUTURE      = 10      # future steps used in the average

# ── Integral correction (eliminates steady-state model-mismatch offset) ───────
# Sign: e0 = la - target; when e0 > 0 persistently, reduce steer → use -I_GAIN
I_GAIN  = 0.020  # per-step integral gain
I_CLIP  = 3.0      # anti-windup clip

# ── Derivative correction (damps error-change oscillation → lower jerk) ───────
# D_GAIN < 0: when e0 rising (la overshooting), reduce steer ✓
D_GAIN  = -0.05    # gain on one-step error change

# ── Output smoothing ──────────────────────────────────────────────────────────
SMOOTH     = 0.87     # EMA weight on previous output  ← best found by sweep
RATE_LIMIT = 0.07     # max steer change per 100 ms step


# ── Precompute all MPC gain vectors (called once) ────────────────────────────
def _build_gains():
    """
    Returns scalar k_e0, vector k_free (H,), scalar k_uprev.

    Per-step usage:
        delta_u0 = k_e0 * e0 + k_free @ free_vec + k_uprev * prev_delta_u
    """
    a  = AR_A
    b  = AR_B_CORRECT

    # Sd[i,j] = a^(i-j)  for j ≤ i, else 0   (lower triangular)
    # Su      = b * Sd
    Sd = np.zeros((H, H), dtype=float)
    for i in range(H):
        for j in range(i + 1):
            Sd[i, j] = a ** (i - j)
    Su = b * Sd

    # Free-response: E_free = Sx * e0
    Sx = np.array([a ** (i + 1) for i in range(H)], dtype=float)   # (H,)

    # Difference matrix for jerk penalty  Δu[k] − Δu[k−1]
    D_mat = np.eye(H, dtype=float) - np.eye(H, k=-1, dtype=float)  # (H,H)

    # Cost matrices
    Q_mat  = Q_TRACK  * np.eye(H, dtype=float)
    R_mat  = R_DELTA  * (D_mat.T @ D_mat) + R_EFFORT * np.eye(H, dtype=float)

    # QP matrix  M = Su^T Q Su + R
    M     = Su.T @ Q_mat @ Su + R_mat
    M_inv = np.linalg.inv(M)          # (H, H)  — precomputed once

    # Gain for initial error:  K_e0 = −M⁻¹ Su^T Q Sx   (H-vector)
    K_e0   = -(M_inv @ Su.T @ Q_mat @ Sx)       # (H,)

    # Gain for free-vec:       K_free = −M⁻¹ Su^T Q Sd  (H×H)
    K_free = -(M_inv @ Su.T @ Q_mat @ Sd)        # (H, H)

    # Gain for prev Δu (jerk):  K_uprev = M⁻¹ R_Δ D^T e₁   (H-vector)
    e1       = np.zeros(H, dtype=float); e1[0] = 1.0
    K_uprev  = M_inv @ (R_DELTA * (D_mat.T @ e1))   # (H,)

    # Only the FIRST element of each H-vector matters (receding horizon)
    return float(K_e0[0]), K_free[0].copy(), float(K_uprev[0])


# ── Controller class ──────────────────────────────────────────────────────────
class Controller(BaseController):
    """
    Incremental error-tracking Koopman MPC.

    FF handles steady state; MPC corrects tracking error only.
    All gain vectors precomputed — per-step cost is one dot product.
    """

    _k_e0, _k_free, _k_uprev = _build_gains()   # class-level constants

    def __init__(self):
        self.prev_output    = 0.0   # smoothed steer command (applied)
        self.prev_delta_u   = 0.0   # Δu from previous step (for jerk penalty)
        self.error_integral = 0.0   # running sum of tracking error
        self.prev_error     = 0.0   # e[t-1] for derivative term
        self._init          = True

    # ─────────────────────────────────────────────────────────────────────────
    def _ff_target(self, target_lataccel, future_plan):
        """Blend current target with near-future average (like my_controller)."""
        if future_plan is not None:
            fut = future_plan.lataccel
            n   = min(FF_N_FUTURE, len(fut))
            if n >= FF_N_FUTURE:
                return ((1.0 - FF_FUTURE_WEIGHT) * target_lataccel
                        + FF_FUTURE_WEIGHT * float(np.mean(fut[:n])))
        return float(target_lataccel)

    # ─────────────────────────────────────────────────────────────────────────
    def _build_free_vec(self, state, future_plan):
        """
        Build free_vec (H,) containing ONLY disturbances (roll + bias).

        Target-change anticipation is INTENTIONALLY excluded: the FF's
        future-blend already handles it, and including it in MPC causes
        Δu spikes that exceed RATE_LIMIT every step, invalidating the
        optimisation.

            free[k] = AR_CROLL * roll[k]  +  AR_D
        """
        roll_seq = np.empty(H, dtype=float)
        roll_seq[0] = float(state.roll_lataccel)
        if future_plan is not None:
            fut_roll = future_plan.roll_lataccel
            n = min(H - 1, len(fut_roll))
            roll_seq[1:n + 1] = fut_roll[:n]
            roll_seq[n + 1:] = (float(fut_roll[-1]) if n > 0 else roll_seq[0])
        else:
            roll_seq[1:] = roll_seq[0]

        return AR_CROLL * roll_seq + AR_D

    # ─────────────────────────────────────────────────────────────────────────
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Warm-start on first call
        if self._init:
            self._init = False
            # Guess prev_output from FF so smoothing starts in a sensible place
            self.prev_output = FF_GAIN * float(target_lataccel)

        # ── 1. Feedforward base ───────────────────────────────────────────────
        eff_tgt   = self._ff_target(target_lataccel, future_plan)
        steer_ff  = FF_GAIN * eff_tgt

        # ── 2. Tracking error + integral + derivative ────────────────────────
        e0 = float(current_lataccel) - float(target_lataccel)
        self.error_integral = float(np.clip(
            self.error_integral + e0, -I_CLIP, I_CLIP))
        d_error = e0 - self.prev_error

        # ── 3. Free disturbance vector (roll + bias only) ────────────────────
        free_vec = self._build_free_vec(state, future_plan)

        # ── 4. Optimal Δu from MPC (proportional + disturbance) ──────────────
        mpc_delta = (self._k_e0 * e0
                     + float(self._k_free @ free_vec)
                     + self._k_uprev * self.prev_delta_u)

        # ── 5. Add integral + derivative corrections ──────────────────────────
        # I: −I_GAIN*integral removes persistent offset
        #    (e0 = la−target > 0 → integral > 0 → −I_GAIN*integral reduces steer ✓)
        # D: D_GAIN < 0 → damps rapid error increases (reduces jerk)
        delta_u0 = mpc_delta - I_GAIN * self.error_integral + D_GAIN * d_error

        # ── 6. Compose total steer, smooth, rate-limit, clip ─────────────────
        raw    = steer_ff + delta_u0
        output = SMOOTH * self.prev_output + (1.0 - SMOOTH) * raw
        output = float(np.clip(output,
                               self.prev_output - RATE_LIMIT,
                               self.prev_output + RATE_LIMIT))
        steer  = float(np.clip(output, -STEER_MAX, STEER_MAX))

        # ── 7. Update state ───────────────────────────────────────────────────
        self.prev_output  = steer
        self.prev_delta_u = steer - steer_ff
        self.prev_error   = e0

        return steer
