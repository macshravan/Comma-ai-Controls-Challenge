import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import glob
import os
from sklearn.metrics import r2_score, mean_squared_error

DATA_DIR   = "./trainingdata"
N_FILES    = 50
STATE_COLS = ["vEgo", "aEgo", "roll", "current_lataccel"]
H_MPC      = 15   # horizon used in MPC — evaluate H-step prediction from true states

x_scaler = joblib.load("x_scaler.pkl")
u_scaler = joblib.load("u_scaler.pkl")
w        = np.load("w.npy")
meta     = np.load("meta.npy")
D_LAT    = int(meta[0])
D_U      = int(meta[1])
D        = max(D_LAT, D_U)


def predict_one(lat_lags, u_lags, x_sc_k):
    ctx = [x_sc_k[0], x_sc_k[1], x_sc_k[2]]
    row = list(lat_lags) + list(u_lags) + ctx + [1.0]
    return float(w @ row)


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))[:N_FILES]
    trajs = []
    for fp in files:
        df = pd.read_csv(fp)
        trajs.append((
            df[STATE_COLS].to_numpy(dtype=np.float64),
            df[["steerCommand"]].to_numpy(dtype=np.float64).flatten(),
            df[["current_lataccel"]].to_numpy(dtype=np.float64).flatten(),
        ))
    return trajs


def evaluate():
    trajs = load_data()

    one_step_preds, one_step_true = [], []
    rollout_preds,  rollout_true  = [], []
    hstep_preds,    hstep_true    = [], []

    for x, u, y in trajs:
        x_sc = x_scaler.transform(x)
        u_sc = u_scaler.transform(u.reshape(-1, 1)).flatten()
        T    = len(y)

        # ── ONE-STEP ────────────────────────────────────────────────────────────
        for k in range(D, T - 1):
            lat_lags = [y[k - d]    for d in range(D_LAT)]
            u_lags   = [u_sc[k - d] for d in range(D_U)]
            one_step_preds.append(predict_one(lat_lags, u_lags, x_sc[k]))
            one_step_true.append(y[k + 1])

        # ── FULL ROLLOUT (uses TRUE context at each step) ───────────────────────
        lat_buf = [y[D-1-d]    for d in range(D_LAT)]
        u_buf   = [u_sc[D-1-d] for d in range(D_U)]
        y_roll  = []
        for k in range(D, T - 1):
            lat_pred = predict_one(lat_buf, u_buf, x_sc[k])
            y_roll.append(lat_pred)
            # shift buffers — use TRUE u, predicted lat
            lat_buf = [lat_pred] + lat_buf[:-1]
            u_buf   = [u_sc[k]]  + u_buf[:-1]
        rollout_preds.extend(y_roll)
        rollout_true.extend(y[D + 1 : T].tolist())

        # ── H-STEP FROM TRUE STATES (MPC-relevant metric) ──────────────────────
        for start in range(D, T - H_MPC - 1, 5):  # every 5 steps
            lat_buf_h = [y[start - d]    for d in range(D_LAT)]
            u_buf_h   = [u_sc[start - d] for d in range(D_U)]
            for h in range(H_MPC):
                lat_pred_h = predict_one(lat_buf_h, u_buf_h, x_sc[start + h])
                hstep_preds.append(lat_pred_h)
                hstep_true.append(y[start + h + 1])
                lat_buf_h = [lat_pred_h]      + lat_buf_h[:-1]
                u_buf_h   = [u_sc[start + h]] + u_buf_h[:-1]

    def rmse(a, b):
        return np.sqrt(mean_squared_error(np.array(a), np.array(b)))

    print("\n=== ONE-STEP (uses true history each step) ===")
    print(f"R2   : {r2_score(one_step_true, one_step_preds):.4f}")
    print(f"RMSE : {rmse(one_step_true, one_step_preds):.4f}")

    print("\n=== FULL ROLLOUT (true context, open-loop lat prediction) ===")
    print(f"R2   : {r2_score(rollout_true, rollout_preds):.4f}")
    print(f"RMSE : {rmse(rollout_true, rollout_preds):.4f}")

    print(f"\n=== H={H_MPC}-STEP FROM TRUE STATES (MPC-relevant) ===")
    print(f"R2   : {r2_score(hstep_true, hstep_preds):.4f}")
    print(f"RMSE : {rmse(hstep_true, hstep_preds):.4f}")

    # plot first trajectory rollout
    traj_len = len(trajs[0][2]) - D - 1
    plt.figure(figsize=(12, 5))
    plt.plot(trajs[0][2][D + 1:], label="True", linewidth=2)
    plt.plot(rollout_preds[:traj_len], label="Rollout (true ctx)", linestyle="--")
    plt.title("Rollout vs Ground Truth (scalar AR, true vEgo/roll)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    evaluate()
