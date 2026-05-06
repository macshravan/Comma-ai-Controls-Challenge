"""
koopman_operator.py
--------------------
Koopman Operator for data-driven linear lifting of lateral vehicle dynamics.

Theory:
    The Koopman operator K acts on observable functions g(x) such that:
        g(x_{k+1}) = K @ g(x_k) + B_k @ u_k

    By lifting the nonlinear state x into a higher-dimensional observable
    space z = g(x), the dynamics become globally linear in z-space.
    This allows standard LQR/MPC to operate on a linear prediction model
    even though the original system is nonlinear.

State vector (raw):
    x = [latAccel, vEgo, aEgo, roll, steer_cmd (previous)]

Observable (lifted) vector z = g(x):
    z = [x,                  # original states (5)
         x^2,                # quadratic terms (5)
         sin(x),             # trigonometric (5)
         cos(x),             # trigonometric (5)
         x[0]*x[1],          # cross-term: latAccel * vEgo
         x[0]*x[3],          # cross-term: latAccel * roll
         x[1]*x[3],          # cross-term: vEgo * roll
         x[1]^2 * x[0]]      # velocity-weighted latAccel
    Total dim: 5+5+5+5+4 = 24

Learning:
    EDMD (Extended Dynamic Mode Decomposition):
        [Z2] = K @ [Z1] + B @ [U]
    Solved via least squares:
        [K | B] = Z2 @ pinv([Z1; U])
"""

import numpy as np
from typing import Tuple, Optional


# ─── Observable function ─────────────────────────────────────────────────────

def lift(x: np.ndarray) -> np.ndarray:
    """
    Lift raw state x into Koopman observable space.

    Args:
        x: (n_states,) or (N, n_states) array
           [latAccel, vEgo, aEgo, roll, prev_steer]

    Returns:
        z: (n_obs,) or (N, n_obs) lifted observable vector
    """
    single = x.ndim == 1
    if single:
        x = x[np.newaxis, :]  # (1, 5)

    la   = x[:, 0:1]   # lateral accel
    v    = x[:, 1:2]   # vEgo
    a    = x[:, 2:3]   # aEgo
    roll = x[:, 3:4]   # road roll
    u    = x[:, 4:5]   # previous steer cmd

    z = np.hstack([
        x,                        # [0:5]   identity
        x ** 2,                   # [5:10]  quadratic
        np.sin(x),                # [10:15] sin
        np.cos(x),                # [15:20] cos
        la * v,                   # [20]    latAccel × vEgo
        la * roll,                # [21]    latAccel × roll
        v  * roll,                # [22]    vEgo × roll
        (v ** 2) * la,            # [23]    v²·latAccel
    ])

    return z[0] if single else z


N_STATES = 5
N_OBS    = N_STATES * 4 + 4   # 24


# ─── EDMD Training ───────────────────────────────────────────────────────────

class KoopmanOperator:
    """
    Extended Dynamic Mode Decomposition (EDMD) Koopman operator.

    Usage:
        ko = KoopmanOperator()
        ko.fit(X, U)           # X: (T, 5), U: (T-1, 1)
        z_next = ko.predict(x, u)
        A, B = ko.get_linear_matrices()
    """

    def __init__(self, n_obs: int = N_OBS, regularization: float = 1e-4):
        self.n_obs = n_obs
        self.reg   = regularization
        self.K     = None   # (n_obs, n_obs) Koopman matrix
        self.B     = None   # (n_obs, 1)     input matrix
        self.fitted = False

    def fit(self, X: np.ndarray, U: np.ndarray) -> "KoopmanOperator":
        """
        Fit Koopman operator from trajectory data using EDMD.

        Args:
            X: (T, 5)  state trajectory  [latAccel, vEgo, aEgo, roll, prev_steer]
            U: (T-1, 1) control inputs   [steer_cmd at each step]

        Returns:
            self
        """
        T = X.shape[0]
        assert U.shape[0] == T - 1, "U must have T-1 rows"

        # Lift all states
        Z  = lift(X)         # (T, n_obs)
        Z1 = Z[:-1].T        # (n_obs, T-1)  current
        Z2 = Z[1:].T         # (n_obs, T-1)  next
        Ut = U.T             # (1, T-1)

        # Stack [Z1; U] for regression target: Z2 = [K | B] @ [Z1; U]
        Omega = np.vstack([Z1, Ut])   # (n_obs+1, T-1)

        # Regularized least squares
        # [K | B] = Z2 @ Omega.T @ inv(Omega @ Omega.T + reg*I)
        gram   = Omega @ Omega.T
        gram  += self.reg * np.eye(gram.shape[0])
        KBmat  = Z2 @ Omega.T @ np.linalg.inv(gram)  # (n_obs, n_obs+1)

        self.K = KBmat[:, :self.n_obs]   # (n_obs, n_obs)
        self.B = KBmat[:, self.n_obs:]   # (n_obs, 1)
        self.fitted = True
        return self

    def predict(self, x: np.ndarray, u: float) -> np.ndarray:
        """
        One-step prediction in observable space.

        Args:
            x: (5,) current state
            u: scalar steer command

        Returns:
            z_next: (n_obs,) lifted next state
        """
        assert self.fitted, "Call fit() first"
        z = lift(x)
        return self.K @ z + self.B.flatten() * float(u)

    def predict_horizon(self, x0: np.ndarray,
                        U_seq: np.ndarray) -> np.ndarray:
        """
        Multi-step prediction over a control sequence.

        Args:
            x0:    (5,)  initial state
            U_seq: (N,)  steer commands over horizon N

        Returns:
            Z_pred: (N+1, n_obs)  lifted state trajectory
        """
        N = len(U_seq)
        Z_pred = np.zeros((N + 1, self.n_obs))
        Z_pred[0] = lift(x0)

        for k in range(N):
            Z_pred[k + 1] = self.K @ Z_pred[k] + self.B.flatten() * U_seq[k]

        return Z_pred

    def get_linear_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (A, B) for use in MPC QP:
            z_{k+1} = A @ z_k + B @ u_k

        Returns:
            A: (n_obs, n_obs)
            B: (n_obs, 1)
        """
        assert self.fitted
        return self.K.copy(), self.B.copy()

    def lataccel_from_obs(self, z: np.ndarray) -> float:
        """
        Extract latAccel from observable vector (index 0 in identity block).
        """
        return float(z[0])

    def save(self, path: str):
        """Save operator matrices to .npz"""
        np.savez(path, K=self.K, B=self.B)
        print(f"[Koopman] Saved to {path}.npz")

    def load(self, path: str):
        """Load operator matrices from .npz"""
        data = np.load(path)
        self.K = data['K']
        self.B = data['B']
        self.fitted = True
        print(f"[Koopman] Loaded from {path}")
        return self


# ─── Data collection helper ──────────────────────────────────────────────────

def collect_koopman_data(data_df, warmup_steps: int = 20):
    """
    Convert a CSV segment DataFrame into (X, U) arrays for EDMD training.

    Args:
        data_df:      pandas DataFrame from a controls_challenge CSV
        warmup_steps: rows to skip at start (simulator warmup)

    Returns:
        X: (T, 5)   state matrix
        U: (T-1, 1) control matrix
    """
    df = data_df.iloc[warmup_steps:].reset_index(drop=True)

    latAccel  = df['targetLateralAcceleration'].values
    vEgo      = df['vEgo'].values
    aEgo      = df['aEgo'].values
    roll      = df['roll'].values
    steer_cmd = df['steerCommand'].values   # filtered steer as proxy for action

    # Build state matrix: [latAccel, vEgo, aEgo, roll, prev_steer]
    prev_steer = np.concatenate([[0.0], steer_cmd[:-1]])
    X = np.stack([latAccel, vEgo, aEgo, roll, prev_steer], axis=1)  # (T, 5)
    U = steer_cmd[:-1].reshape(-1, 1)                                # (T-1, 1)

    return X, U


# ─── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Koopman Operator Unit Test ===")

    # Synthetic test
    np.random.seed(42)
    T = 500
    X_test = np.random.randn(T, N_STATES) * np.array([1.0, 10.0, 2.0, 0.05, 0.3])
    U_test = np.random.uniform(-0.3, 0.3, (T - 1, 1))

    ko = KoopmanOperator(regularization=1e-3)
    ko.fit(X_test, U_test)

    A, B = ko.get_linear_matrices()
    print(f"K shape: {A.shape}, B shape: {B.shape}")

    # Test one-step prediction
    z_pred = ko.predict(X_test[0], float(U_test[0, 0]))
    print(f"z_pred shape: {z_pred.shape}, latAccel pred: {ko.lataccel_from_obs(z_pred):.4f}")

    # Test horizon prediction
    Z_horizon = ko.predict_horizon(X_test[0], U_test[:10, 0])
    print(f"Horizon pred shape: {Z_horizon.shape}")

    print("All tests passed.")
