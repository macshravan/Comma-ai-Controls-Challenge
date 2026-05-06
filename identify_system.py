"""
System Identification for Controls Challenge
=============================================
Runs the PID controller on N route files, collects real (state, steer, next_state)
transitions from the simulator, then fits three models:

  1. AR model     : la(t+1) = a*la(t) + b*steer(t) + c*roll(t) + d
                   Simple autoregressive model. Used by MPC for predictions.

  2. LQR system   : State x=[error, error_integral], system matrices A, B
                   Gain K computed from the Discrete Algebraic Riccati Equation.

  3. Koopman model: z(t+1) = A_K @ z(t) + B_K * steer(t)
                   State lifted to 11D polynomial observable space via EDMD.
                   Captures nonlinear car dynamics as a linear system.
                   LQR gain K_koopman computed in lifted space.

All models saved to models/system_models.npz
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import linalg
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tinyphysics import (TinyPhysicsModel, TinyPhysicsSimulator,
                         CONTROL_START_IDX, COST_END_IDX, CONTEXT_LENGTH)

MODEL_PATH = './models/tinyphysics.onnx'
DATA_PATH  = './SYNTHETIC'
N_FILES    = 50      # files used for identification (more = better fit, slower)
OBS_DIM    = 11      # dimension of Koopman observable space


# ─────────────────────────────────────────────────────────────────────────────
# Koopman observable (lifting function)
# Takes a 4D state and maps to 11D feature space including polynomial terms.
# Key property: the nonlinear system becomes approximately linear in this space.
# ─────────────────────────────────────────────────────────────────────────────
def koopman_obs(la, error, roll, v_ego):
    la    = np.clip(la,    -5.0, 5.0)
    error = np.clip(error, -5.0, 5.0)
    roll  = np.clip(roll,  -2.0, 2.0)
    v_n   = (v_ego - 15.0) / 10.0   # normalize: 15 m/s is typical highway speed
    return np.array([
        la, error, roll, v_n,        # 4 linear terms  (original state)
        la**2, error**2,             # 2 quadratic terms (captures nonlinearity)
        la * error, la * roll,       # 2 cross terms
        error * roll, error * v_n,   # 2 cross terms (speed-dependent behavior)
        1.0                          # 1 bias term
    ])  # total: 11


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Collect transitions by running full simulations
# ─────────────────────────────────────────────────────────────────────────────
def collect_transitions(data_files, physics_model, n_files=50):
    """
    Run PID on each file, then extract (state, action, next_state) transitions
    from the CONTROL phase (steps 100-500) where the controller is active.

    We use the CONTROL phase because:
    - Pre-control phase (steps 0-99) just replays recorded steering, current_lataccel
      equals target_lataccel (no useful error signal for identification)
    - Control phase has real deviations from target that reveal the system dynamics
    """
    from controllers.pid import Controller as PIDController

    la, next_la, steer = [], [], []
    roll, v_ego, target = [], [], []

    print(f"Collecting transitions from {n_files} simulations...")

    for i, f in enumerate(data_files[:n_files]):
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_files}] processed...")

        pid = PIDController()
        sim = TinyPhysicsSimulator(physics_model, str(f), controller=pid, debug=False)
        sim.rollout()

        # Extract transitions from the control phase
        for t in range(CONTROL_START_IDX, min(COST_END_IDX, len(sim.current_lataccel_history) - 1)):
            la.append(sim.current_lataccel_history[t])
            next_la.append(sim.current_lataccel_history[t + 1])
            steer.append(sim.action_history[t])
            roll.append(sim.state_history[t].roll_lataccel)
            v_ego.append(sim.state_history[t].v_ego)
            target.append(sim.target_lataccel_history[t])

    data = {
        'la':      np.array(la),
        'next_la': np.array(next_la),
        'steer':   np.array(steer),
        'roll':    np.array(roll),
        'v_ego':   np.array(v_ego),
        'target':  np.array(target),
    }
    data['error'] = data['target'] - data['la']
    print(f"  Collected {len(la)} transitions total.\n")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fit AR model  la(t+1) = a*la + b*steer + c*roll + d
# ─────────────────────────────────────────────────────────────────────────────
def fit_ar_model(data):
    """
    Simple autoregressive (ARX) model.
    Regressor: [la(t), steer(t), roll(t), 1]
    Solved with ordinary least squares.
    R² tells us how much variance is explained (1.0 = perfect).
    """
    Phi = np.column_stack([data['la'], data['steer'], data['roll'], np.ones(len(data['la']))])
    theta, _, _, _ = np.linalg.lstsq(Phi, data['next_la'], rcond=None)
    a, b, c, d = theta

    pred   = Phi @ theta
    resid  = data['next_la'] - pred
    r2     = 1 - np.var(resid) / np.var(data['next_la'])

    print(f"AR Model R² = {r2:.4f}")
    print(f"  la(t+1) = {a:.4f}·la + {b:.4f}·steer + {c:.4f}·roll + {d:.6f}")
    return {'a': a, 'b': b, 'c': c, 'd': d}


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build LQR state-space and compute optimal gain K
# ─────────────────────────────────────────────────────────────────────────────
def build_and_solve_lqr(ar_params):
    """
    State: x = [error, d_error, error_integral]  (3D — includes derivative for damping)
    where error = la - target

    AR model: la(t+1) = a*la + b*steer + c*roll + d
    In error coordinates (e = la - target):
      e(t+1)    ≈  a*e + b*steer + disturbance
      de(t+1)   ≈  e(t+1) - e(t) = (a-1)*e + b*steer + ...
      eint(t+1) =  eint + e

    The derivative state adds damping — analogous to the D term in PID.

    A = [[a,   0,  0],   B = [[b],
         [a-1, 0,  0],        [b],
         [1,   0,  1]]        [0]]

    Q and R scaled so that K ≈ PID-like gains:
      Q[0,0] = 50 (error), Q[1,1] = 5 (derivative damping), Q[2,2] = 0.1 (integral)
      R = 2000  → gain magnitude comparable to PID p=0.2
    """
    a, b = ar_params['a'], ar_params['b']
    dt = 1.0  # 100ms steps

    A = np.array([[a,       0.0, 0.0],
                  [a - 1.0, 0.0, 0.0],
                  [dt,      0.0, 1.0]])
    B = np.array([[b],
                  [b],
                  [0.0]])
    Q = np.diag([50.0, 5.0, 0.1])
    R = np.array([[2000.0]])

    try:
        P   = linalg.solve_discrete_are(A, B, Q, R)
        K   = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        A_cl = A - B @ K
        max_eig = np.max(np.abs(np.linalg.eigvals(A_cl)))
        print(f"LQR gain K = {K}")
        print(f"Closed-loop max eigenvalue = {max_eig:.4f} ({'stable ✓' if max_eig < 1 else 'UNSTABLE ✗'})")
        return A, B, K
    except Exception as e:
        print(f"LQR DARE failed: {e}. Using fallback gains.")
        K_fallback = np.array([[0.25, 0.05, 0.08]])
        return A, B, K_fallback


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Koopman EDMD — fit A_K and B_K in lifted space
# ─────────────────────────────────────────────────────────────────────────────
def fit_koopman_model(data):
    """
    Extended Dynamic Mode Decomposition (EDMD).

    For each transition (x_t, u_t) -> x_{t+1}:
      z_t  = g(x_t)    # lift to observable space
      z_t1 = g(x_{t+1})

    Solve:  [A_K | B_K] = Z' · Ω^T · (Ω·Ω^T + λI)^{-1}
    where   Ω = [Z; U]  (stacked observables + actions)

    This gives a LINEAR system in z-space:
      z(t+1) = A_K · z(t) + B_K · u(t)

    The key insight: nonlinear car dynamics become approximately linear
    when viewed through the right set of observable functions.
    """
    T    = len(data['la'])
    error = data['target'] - data['la']

    print(f"Fitting Koopman model on {T} transitions (EDMD)...")

    # Build observable matrices: shape (OBS_DIM, T)
    Z      = np.zeros((OBS_DIM, T))
    Z_next = np.zeros((OBS_DIM, T))

    for t in range(T):
        next_err = data['target'][t] - data['next_la'][t]  # approximate
        Z[:, t]      = koopman_obs(data['la'][t],      error[t],   data['roll'][t], data['v_ego'][t])
        Z_next[:, t] = koopman_obs(data['next_la'][t], next_err,   data['roll'][t], data['v_ego'][t])

    U   = data['steer'].reshape(1, T)
    Omega = np.vstack([Z, U])          # shape: (OBS_DIM+1, T)

    # Tikhonov regularization prevents overfitting in noisy data
    lam  = 1e-3
    G    = Omega @ Omega.T + lam * np.eye(Omega.shape[0])
    A_k  = Z_next @ Omega.T @ np.linalg.inv(G)

    A_K  = A_k[:, :OBS_DIM]           # (OBS_DIM × OBS_DIM)
    B_K  = A_k[:, OBS_DIM:].reshape(-1, 1)  # (OBS_DIM × 1)

    # Validation: predict la on a held-out slice
    n_val = min(500, T)
    Z_pred = A_K @ Z[:, :n_val] + B_K * U[:, :n_val]
    pred_la   = Z_pred[0, :]
    actual_la = data['next_la'][:n_val]
    r2 = 1 - np.var(pred_la - actual_la) / np.var(actual_la)
    print(f"Koopman model R² (la prediction) = {r2:.4f}")

    # Check stability
    eigvals  = np.linalg.eigvals(A_K)
    max_eig  = np.max(np.abs(eigvals))
    print(f"A_K max eigenvalue magnitude = {max_eig:.4f}")

    # Soft stabilization: if highly unstable, scale down
    if max_eig > 2.0:
        print(f"  Stabilizing A_K (dividing by {max_eig:.2f})...")
        A_K = A_K / max_eig
        print(f"  New max eigenvalue = {np.max(np.abs(np.linalg.eigvals(A_K))):.4f}")

    return A_K, B_K


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: LQR in Koopman lifted space
# ─────────────────────────────────────────────────────────────────────────────
def compute_koopman_lqr(A_K, B_K):
    """
    Apply LQR to the lifted Koopman system.

    The cost penalizes the FIRST observable (la) tracking the target.
    C extracts la from z:  la = C @ z   where C = [1, 0, 0, ..., 0]

    Q_K = 50 · C^T C   (project tracking cost into lifted space)
    R   = 1.0

    The result K_koopman has shape (1, OBS_DIM).
    At runtime:  u = -K_koopman @ (z - z_target)
    """
    n   = A_K.shape[0]
    C   = np.zeros((1, n));  C[0, 0] = 1.0
    Q   = 50.0 * C.T @ C
    R   = np.array([[100.0]])  # larger R = smaller gains = less saturation

    try:
        P   = linalg.solve_discrete_are(A_K, B_K, Q, R)
        K   = np.linalg.inv(R + B_K.T @ P @ B_K) @ B_K.T @ P @ A_K

        A_cl = A_K - B_K @ K
        max_eig = np.max(np.abs(np.linalg.eigvals(A_cl)))
        print(f"Koopman LQR: closed-loop max eigenvalue = {max_eig:.4f} "
              f"({'stable ✓' if max_eig < 1 else 'UNSTABLE ✗'})")
        return K
    except Exception as e:
        print(f"Koopman LQR DARE failed: {e}")
        # Fallback: proportional gain in lifted space
        K_fb = np.zeros((1, n));  K_fb[0, 0] = 0.3
        return K_fb


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  System Identification — Controls Challenge")
    print("=" * 60)

    data_files    = sorted(Path(DATA_PATH).iterdir())[:N_FILES]
    physics_model = TinyPhysicsModel(MODEL_PATH, debug=False)

    # 1. Collect transitions
    data = collect_transitions(data_files, physics_model, N_FILES)

    # 2. AR model
    print("--- AR Model ---")
    ar = fit_ar_model(data)

    # 3. LQR
    print("\n--- LQR ---")
    A_lqr, B_lqr, K_lqr = build_and_solve_lqr(ar)

    # 4. Koopman EDMD
    print("\n--- Koopman EDMD ---")
    A_K, B_K = fit_koopman_model(data)

    # 5. Koopman LQR
    print("\n--- Koopman LQR ---")
    K_koopman = compute_koopman_lqr(A_K, B_K)

    # 6. Save everything
    Path('models').mkdir(exist_ok=True)
    np.savez('models/system_models.npz',
             ar_a=ar['a'], ar_b=ar['b'], ar_c=ar['c'], ar_d=ar['d'],
             A_lqr=A_lqr, B_lqr=B_lqr, K_lqr=K_lqr,
             A_K=A_K, B_K=B_K, K_koopman=K_koopman)

    print("\n✓ All models saved to models/system_models.npz")
    print("\nSummary:")
    print(f"  AR:      a={ar['a']:.4f}, b={ar['b']:.4f}, c={ar['c']:.4f}, d={ar['d']:.6f}")
    print(f"  LQR K  = {K_lqr}")
    print(f"  Koopman: A_K {A_K.shape}, B_K {B_K.shape}")
    print(f"  Koopman LQR K shape = {K_koopman.shape}")
