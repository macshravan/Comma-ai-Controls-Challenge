# Comma AI Controls Challenge

A project exploring advanced control strategies for lateral acceleration tracking in autonomous driving, built on top of Comma AI's [Controls Challenge](https://github.com/commaai/controls_challenge) simulation framework.

The goal is to design controllers that minimize a cost function balancing **lateral acceleration tracking error** and **steering jerk**, evaluated against a simulated vehicle physics model (`tinyphysics.onnx`).

---

## Project Structure

```
.
├── tinyphysics.py          # Core simulator: TinyPhysicsModel + TinyPhysicsSimulator
├── eval.py                 # Batch evaluation & HTML report generation
├── identify_system.py      # System identification: AR, LQR, and Koopman models
├── koopman_operator.py     # Koopman observable lifting functions (EDMD)
├── koopman_trainingv3.py   # Koopman model training pipeline
├── koopman_testing.py      # Koopman model validation
├── experiment.ipynb        # Experiments & analysis notebook
├── models/
│   └── tinyphysics.onnx    # ONNX vehicle physics model
├── Controllers/
│   ├── __init__.py         # BaseController interface
│   ├── pid.py              # Baseline PID controller
│   ├── feedforward_pid.py  # Feedforward + PID with future preview
│   ├── ar_mpc_model.py     # AR(1)-based incremental MPC
│   ├── koopman_mpc.py      # Koopman operator + MPC controller
│   ├── koopman_mpc_jd.py   # Earlier Koopman MPC implementation (superseded)
│   ├── zero.py             # Zero-output controller (sanity check)
│   └── log_inputs.py       # Input logging controller
├── A.npy, B.npy, C.npy    # Saved Koopman system matrices
├── koopman_trained.npz     # Trained Koopman model weights
├── phi.pkl                 # Koopman feature map (RBFSampler)
├── x_scaler.pkl            # State scaler for Koopman input
├── u_scaler.pkl            # Action scaler for Koopman input
├── koopman_cfg.pkl         # Koopman hyperparameter config
└── requirements.txt        # Python dependencies
```

---

## Cost Function

Controllers are scored on two metrics computed over simulation steps 100–500:

- **Lateral Acceleration Cost**: Mean squared error between target and actual lateral acceleration, scaled by ×100
- **Jerk Cost**: Mean squared rate of change of lateral acceleration (smoothness), scaled by ×100
- **Total Cost**: `50 × lataccel_cost + jerk_cost`

Lower is better. The baseline PID controller serves as the reference to beat.

---

## Controllers

### 1. PID (`Controllers/pid.py`)
The baseline controller. A simple proportional-integral-derivative controller with hand-tuned gains:
- `P = 0.195`, `I = 0.100`, `D = -0.053`

### 2. Feedforward PID (`Controllers/feedforward_pid.py`)
An improved controller over the baseline with several key additions:
- **Feedforward term**: Uses an empirically derived steady-state gain (`FF_GAIN ≈ 0.315`) to eliminate tracking lag at constant targets
- **Future preview**: Blends current target (30%) with an average of the next 10 future targets (70%) to anticipate upcoming turns
- **Integral anti-windup**: Clips the integral term to `[-5, 5]`
- **Rate limiting + smoothing**: Caps steer change per step and applies exponential smoothing to reduce jerk

### 3. AR-MPC (`Controllers/ar_mpc_model.py`)
An incremental MPC controller built on an AR(1) lateral dynamics model identified from data:

```
la(t+1) = 0.9577·la(t) + 0.0605·steer(t) + 0.0404·roll(t) - 0.0078
```

Key design choices:
- The MPC operates only in **error space** (`e = la - target`), not full target tracking, to avoid model mismatch-induced oscillation
- Feedforward handles steady-state; MPC corrects only residual disturbances
- Closed-form optimal gain precomputed from the discrete Riccati equation (no online optimization)

### 4. Koopman MPC (`Controllers/koopman_mpc.py`)
The most advanced controller, using **Koopman operator theory** to lift the nonlinear vehicle dynamics into a higher-dimensional linear space where standard MPC applies:

- **Koopman lifting**: State `[v_ego, a_ego, roll_lataccel, current_lataccel]` is lifted via a trained feature map (`phi.pkl`) to a higher-dimensional observable space
- **Linear prediction**: `z(t+1) = A·z(t) + B·u(t)` in lifted space (matrices saved as `A.npy`, `B.npy`)
- **Sampling-based MPC**: Evaluates `N_CANDIDATES = 30` candidate steer values over horizon `H = 8`, selecting the one minimizing a cost weighted by tracking error and jerk
- **Hybrid control**: Falls back to feedforward + PID when the Koopman state buffer is not yet full (warm-up period)

---

## System Identification (`identify_system.py`)

Runs the PID controller on `N = 50` simulation files to collect real `(state, steer, next_state)` transitions, then fits three models:

1. **AR Model** — simple autoregressive fit via ordinary least squares
2. **LQR** — discrete-time LQR computed from the Discrete Algebraic Riccati Equation (DARE) on a 3-state error system `[error, d_error, error_integral]`
3. **Koopman EDMD** — Extended Dynamic Mode Decomposition on an 11-dimensional polynomial observable space capturing nonlinear dynamics as a linear system

---

## Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Download dataset:**
The dataset (~0.6 GB) is hosted on HuggingFace. It will auto-download when you first run the simulator:
```bash
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --controller pid
```

Or download manually:
```
https://huggingface.co/datasets/commaai/commaSteeringControl
```

---

## Usage

### Run a single rollout (with debug plots)
```bash
python tinyphysics.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data/00000.csv \
  --controller feedforward_pid \
  --debug
```

### Evaluate over multiple segments
```bash
python tinyphysics.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 100 \
  --controller koopman_mpc
```

### Generate comparison report (test vs baseline)
```bash
python eval.py \
  --model_path ./models/tinyphysics.onnx \
  --data_path ./data \
  --num_segs 100 \
  --test_controller feedforward_pid \
  --baseline_controller pid
```
This produces `report.html` with cost distribution plots and sample rollout visualizations.

### Run system identification
```bash
python identify_system.py
```
This re-fits all system models and saves them to `models/system_models.npz`.

---

## Available Controllers

| Name | Description |
|---|---|
| `pid` | Baseline PID |
| `feedforward_pid` | Feedforward + PID with future preview |
| `ar_mpc_model` | AR(1) incremental MPC |
| `koopman_mpc` | Koopman operator + sampling MPC |
| `koopman_mpc_jd` | Earlier Koopman MPC implementation (superseded by `koopman_mpc`) |
| `zero` | Zero output (sanity check) |
| `log_inputs` | Logs controller inputs for analysis |

---

## Results

Sample rollout plots for each controller are saved as:
- `feedforward_pid.png` / `feedforward_pid_debug.png`
- `ar_mpc_model.png` / `ar_mpc_model_debug.png`
- `koopman_mpc.png` / `koopman_mpc_debug.png`
- `koopma_jd.png` / `koopma_jd_debug.png`


---

## License

See [LICENSE](LICENSE).
