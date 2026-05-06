"""
Koopman Operator Model — comma.ai Controls Challenge
======================================================
Delta-blended Koopman with delay embedding + polynomial lifting.
Trains A, B, C matrices for use in an MPC steer controller.
"""

import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

# =========================
# CONFIG
# =========================
STATE_COLS  = ["vEgo", "aEgo", "roll", "current_lataccel"]
DATA_DIR    = "./trainingdata"

DELAY       = 3          # delay embedding window (more temporal context)
POLY_DEG    = 2          # polynomial lift degree
RIDGE       = 0.05       # ridge regularisation

N_TRAIN     = 900
N_VAL       = 100

MAX_SPECTRAL_RADIUS = 0.97   # tighter than 0.99 → safer rollout

BLEND_ALPHA = 0.80       # weight on delta prediction vs absolute
HORIZON     = 15         # MPC evaluation horizon


# =========================
# DATA LOADING
# =========================
def load_data(data_dir: str, start: int, n: int):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))[start : start + n]
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir} at offset {start}")

    trajs, trajs_u, trajs_y = [], [], []
    for fp in files:
        df = pd.read_csv(fp)
        trajs.append(df[STATE_COLS].to_numpy(dtype=np.float64))
        trajs_u.append(df[["steerCommand"]].to_numpy(dtype=np.float64))
        trajs_y.append(df[["current_lataccel"]].to_numpy(dtype=np.float64))

    print(f"  Loaded {len(files)} files from offset {start}")
    return trajs, trajs_u, trajs_y


# =========================
# DELAY EMBEDDING
# =========================
def delay_embed(x: np.ndarray, delay: int) -> np.ndarray:
    """Stack the last `delay` frames as a single feature vector."""
    return np.array(
        [x[k - delay + 1 : k + 1][::-1].reshape(-1) for k in range(delay - 1, len(x))]
    )


# =========================
# BUILD TRAINING PAIRS
# =========================
def build_pairs(trajs_z, trajs_u_sc, trajs_y):
    Zk, Uk, Zkp1, Y_abs, Y_delta = [], [], [], [], []

    for z, u, y in zip(trajs_z, trajs_u_sc, trajs_y):
        y_al  = y[DELAY - 1 :]          # align to lifted state
        delta = np.diff(y_al, axis=0)   # Δy_t = y_{t+1} - y_t

        Zk.append(z[:-1])
        Zkp1.append(z[1:])
        Uk.append(u[DELAY - 1 : -1])
        Y_abs.append(y_al[:-1])
        Y_delta.append(delta)

    return (
        np.vstack(Zk),
        np.vstack(Uk),
        np.vstack(Zkp1),
        np.vstack(Y_abs),
        np.vstack(Y_delta),
    )


# =========================
# SOLVERS
# =========================
def ridge_solve(X: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + ridge * np.eye(d), X.T @ Y)


def fit_AB(Zk: np.ndarray, Uk: np.ndarray, Zkp1: np.ndarray):
    """Fit linear dynamics:  z_{t+1} = A z_t + B u_t"""
    X = np.hstack([Zk, Uk])
    K = ridge_solve(X, Zkp1, RIDGE)
    nz = Zk.shape[1]
    A  = K[:nz].T
    B  = K[nz:].T
    return A, B


def fit_C_separate(Zk: np.ndarray, Y_abs: np.ndarray, Y_delta: np.ndarray):
    """
    Fit two independent readout heads:
      C[0]: z -> y_abs
      C[1]: z -> Δy
    Fitting separately avoids interference between heads.
    """
    W_abs   = ridge_solve(Zk, Y_abs,   RIDGE)   # (nz, 1)
    W_delta = ridge_solve(Zk, Y_delta, RIDGE)   # (nz, 1)
    return np.vstack([W_abs.T, W_delta.T])       # (2, nz)


def stabilize(A: np.ndarray, max_r: float = MAX_SPECTRAL_RADIUS) -> np.ndarray:
    """Clip eigenvalue magnitudes to enforce spectral radius ≤ max_r."""
    eigs, V = np.linalg.eig(A)
    mags    = np.abs(eigs)
    n_clip  = int((mags > max_r).sum())
    scale   = np.where(mags > max_r, max_r / mags, 1.0)
    A_stab  = (V @ np.diag(eigs * scale) @ np.linalg.inv(V)).real
    print(f"  Stabilise: clipped {n_clip}/{len(eigs)} eigenvalues  "
          f"(max |λ| before={mags.max():.4f}, after={np.abs(np.linalg.eigvals(A_stab)).max():.4f})")
    return A_stab


# =========================
# HORIZON EVALUATION
# =========================
def evaluate_horizon(trajs_x, trajs_u, trajs_y,
                     x_sc, u_sc, phi, A, B, C,
                     H: int = HORIZON,
                     alpha: float = BLEND_ALPHA):
    """
    Open-loop H-step rollout using the recorded control sequence.
    Blended prediction:  ŷ_{t+1} = α*(ŷ_t + C[1]·z) + (1-α)*(C[0]·z)
    """
    all_true, all_pred = [], []

    for x, u, y in zip(trajs_x, trajs_u, trajs_y):
        xs = x_sc.transform(x)
        us = u_sc.transform(u)

        z_dl  = delay_embed(xs, DELAY)
        z     = phi.transform(z_dl)
        y_al  = y[DELAY - 1 :]
        uk    = us[DELAY - 1 :]

        for t in range(len(z) - H - 1):
            z_r = z[t].copy()
            y_r = float(y_al[t, 0])
            preds = []

            for k in range(H):
                y_abs_pred = float(C[0] @ z_r)
                dy_pred    = float(C[1] @ z_r)
                y_r        = alpha * (y_r + dy_pred) + (1.0 - alpha) * y_abs_pred
                preds.append(y_r)
                z_r = A @ z_r + B @ uk[t + k]

            all_pred.append(preds)
            all_true.append(y_al[t + 1 : t + H + 1].flatten())

    return r2_score(np.concatenate(all_true), np.concatenate(all_pred))


# =========================
# MAIN
# =========================
def main():
    print("=" * 50)
    print("  KOOPMAN TRAINING  —  comma.ai Controls Challenge")
    print("=" * 50)

    # ---- load -------------------------------------------------------
    print("\n[1/5] Loading data …")
    trajs,   trajs_u,   trajs_y   = load_data(DATA_DIR, 0,       N_TRAIN)
    trajs_v, trajs_u_v, trajs_y_v = load_data(DATA_DIR, N_TRAIN, N_VAL)

    # ---- scale -------------------------------------------------------
    print("[2/5] Fitting scalers …")
    x_sc = StandardScaler().fit(np.vstack(trajs))
    u_sc = StandardScaler().fit(np.vstack(trajs_u))

    trajs_sc   = [x_sc.transform(x) for x in trajs]
    trajs_u_sc = [u_sc.transform(u) for u in trajs_u]

    # ---- lift -------------------------------------------------------
    print("[3/5] Lifting state space …")
    trajs_dl = [delay_embed(x, DELAY) for x in trajs_sc]

    phi = PolynomialFeatures(degree=POLY_DEG, include_bias=False)
    phi.fit(np.vstack(trajs_dl))

    trajs_z = [phi.transform(x) for x in trajs_dl]
    nz      = trajs_z[0].shape[1]
    print(f"  Lifted dim: {nz}  (delay={DELAY}, poly_deg={POLY_DEG})")

    # ---- build training pairs ----------------------------------------
    print("[4/5] Fitting A, B, C …")
    Zk, Uk, Zkp1, Y_abs, Y_delta = build_pairs(trajs_z, trajs_u_sc, trajs_y)
    print(f"  Training pairs: {Zk.shape[0]:,}")

    A, B = fit_AB(Zk, Uk, Zkp1)
    C    = fit_C_separate(Zk, Y_abs, Y_delta)
    A    = stabilize(A)

    # ---- one-step residuals ------------------------------------------
    Zkp1_hat = (A @ Zk.T + B @ Uk.T).T
    ss_res   = np.sum((Zkp1 - Zkp1_hat) ** 2)
    ss_tot   = np.sum((Zkp1 - Zkp1.mean(0)) ** 2)
    print(f"  One-step z R²: {1 - ss_res/ss_tot:.4f}")

    # ---- horizon eval ------------------------------------------------
    print(f"[5/5] Evaluating H={HORIZON} rollout …")
    r2_train = evaluate_horizon(trajs,   trajs_u,   trajs_y,
                                x_sc, u_sc, phi, A, B, C)
    r2_val   = evaluate_horizon(trajs_v, trajs_u_v, trajs_y_v,
                                x_sc, u_sc, phi, A, B, C)

    print(f"\n  ┌─────────────────────────┐")
    print(f"  │  MPC H={HORIZON} R²  (blend α={BLEND_ALPHA}) │")
    print(f"  │  Train : {r2_train:.4f}           │")
    print(f"  │  Val   : {r2_val:.4f}           │")
    print(f"  └─────────────────────────┘")

    # ---- save --------------------------------------------------------
    joblib.dump(x_sc, "x_scaler.pkl")
    joblib.dump(u_sc, "u_scaler.pkl")
    joblib.dump(phi,  "phi.pkl")
    np.save("A.npy", A)
    np.save("B.npy", B)
    np.save("C.npy", C)

    # Save config so controller can reload without hardcoding
    cfg = dict(DELAY=DELAY, POLY_DEG=POLY_DEG, BLEND_ALPHA=BLEND_ALPHA,
               STATE_COLS=STATE_COLS)
    joblib.dump(cfg, "koopman_cfg.pkl")

    print("\nSaved: A.npy  B.npy  C.npy  x_scaler.pkl  u_scaler.pkl  phi.pkl  koopman_cfg.pkl")
    print("Done ✓")


if __name__ == "__main__":
    main()
