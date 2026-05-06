from . import BaseController
import numpy as np
import csv
import os
import atexit

class Controller(BaseController):
    def __init__(self, data_path=None, log_dir="./trainingdata", **kwargs):
        # PID params
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1

        self.error_integral = 0.0
        self.prev_error = 0.0
        self.step_idx = 0

        # 🔥 Create directory
        os.makedirs(log_dir, exist_ok=True)

        # 🔥 Use input filename → output filename
        if data_path is not None:
            base = os.path.basename(data_path).replace(".csv", "")
            log_path = os.path.join(log_dir, f"{base}.csv")
        else:
            # fallback (rare case)
            import time
            unique_id = f"{int(time.time()*1000)}_{os.getpid()}"
            log_path = os.path.join(log_dir, f"{unique_id}.csv")

        print(f"[LOG] Writing to: {log_path}")

        self.f = open(log_path, "w", newline="", buffering=1)
        self.writer = csv.writer(self.f)

        # Header
        self.writer.writerow([
            "step",
            "target_lataccel",
            "current_lataccel",
            "vEgo",
            "aEgo",
            "roll",
            "steerCommand",
            "future_plan_lataccel",
        ])

        atexit.register(self.close)

    def close(self):
        try:
            if hasattr(self, "f") and not self.f.closed:
                self.f.flush()
                self.f.close()
        except Exception:
            pass

    def _get_state_value(self, state, names, default=np.nan):
        if isinstance(state, dict):
            for n in names:
                if n in state:
                    try:
                        return float(state[n])
                    except Exception:
                        pass

        for n in names:
            if hasattr(state, n):
                try:
                    return float(getattr(state, n))
                except Exception:
                    pass

        return default

    def _get_future_lataccel(self, future_plan):
        if future_plan is None:
            return ""

        if hasattr(future_plan, "lataccel"):
            try:
                return list(np.asarray(future_plan.lataccel, dtype=float).ravel())
            except Exception:
                return str(future_plan.lataccel)

        if isinstance(future_plan, dict) and "lataccel" in future_plan:
            try:
                return list(np.asarray(future_plan["lataccel"], dtype=float).ravel())
            except Exception:
                return str(future_plan["lataccel"])

        return str(future_plan)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # PID
        error = target_lataccel - current_lataccel

        self.error_integral += error
        self.error_integral = np.clip(self.error_integral, -3, 3)

        error_diff = error - self.prev_error
        self.prev_error = error

        steer = self.p * error + self.i * self.error_integral + self.d * error_diff

        # State extraction
        vEgo = self._get_state_value(state, ["vEgo", "v_ego", "speed", "velocity"])
        aEgo = self._get_state_value(state, ["aEgo", "a_ego"])
        roll = self._get_state_value(state, ["roll", "roll_lataccel"])

        future_lataccel = self._get_future_lataccel(future_plan)

        # Write row
        self.writer.writerow([
            self.step_idx,
            float(target_lataccel),
            float(current_lataccel),
            float(vEgo) if not np.isnan(vEgo) else "",
            float(aEgo) if not np.isnan(aEgo) else "",
            float(roll) if not np.isnan(roll) else "",
            float(steer),
            future_lataccel,
        ])

        self.step_idx += 1
        return steer