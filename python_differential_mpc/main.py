import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import time

from msd import MassSpringDamper
from offline_koopman import KoopmanEDMD
from online_mpc import DifferentialKoopmanMPC

def main():
    # 1. Generate Data (if needed)
    print("Checking data...")
    if not os.path.exists('python_differential_mpc/data/X.csv'):
        print("Data not found. Please run generate_data.py first.")
        return

    # 2. Offline Training
    print("Fitting Offline Koopman...")
    koopman = KoopmanEDMD()
    koopman.fit('python_differential_mpc/data/X.csv', 'python_differential_mpc/data/U.csv')
    A0, B0, C0 = koopman.get_matrices()

    # 3. Setup Simulation
    dt = 0.01
    sim_time = 30.0 # seconds
    steps = int(sim_time / dt)
    update_interval = 10 # Control update interval (steps)

    msd = MassSpringDamper(dt=dt)

    # Controller
    # Horizon 50 steps (0.5s)
    mpc = DifferentialKoopmanMPC(A0, B0, C0, horizon=50, dt=dt)

    # Simulation Loop
    x = np.array([1.0, 0.0]) # Initial state
    z = koopman.lift(x).flatten()

    history_z = []
    history_u = []
    history_len = 50 # Window size for adaptation

    # Logs
    log_t = []
    log_x = []
    log_u = []
    log_compute_time = []

    current_u = np.zeros(1)

    print("Starting Closed-Loop Simulation...")

    for k in range(steps):
        t = k * dt

        # Disturbance: Kick at t=15s (index 1500)
        if k == 1500:
             x[1] += 2.0 # Velocity kick
             print(f"Disturbance applied at t={t}")

        # Process Noise
        x += np.random.normal(0, 1e-4, size=x.shape) # Reduced noise

        # Lift
        z = koopman.lift(x).flatten()

        # Store history
        history_z.append(z)
        # We need (z_k, u_k) -> z_{k+1}
        # history_u stores u applied at k

        if len(history_z) > history_len + 1:
            history_z.pop(0)
        if len(history_u) > history_len:
            history_u.pop(0)

        # Control Update
        if k % update_interval == 0:
            start_time = time.time()

            # Shift Plan
            mpc.shift_plan(shift_steps=update_interval)

            # Optimization
            # We need pairs of z_k, u_k to predict z_{k+1}
            # history_z has length L+1, history_u has length L
            if len(history_u) >= 10:
                 # Use recent history for adaptation
                 loss, l_m, l_p = mpc.optimize_step(history_z, history_u, z, steps=5)

            current_u = mpc.get_control()

            # computation time
            comp_time = time.time() - start_time
            log_compute_time.append(comp_time)
        else:
             # Just replicate last computation time for plotting or 0
             log_compute_time.append(log_compute_time[-1] if len(log_compute_time)>0 else 0)

        # Apply Control
        current_u = np.clip(current_u, -20, 20)

        next_x = msd.step(x, current_u)

        # Log
        log_t.append(t)
        log_x.append(x)
        log_u.append(current_u)

        # Update history u
        history_u.append(current_u)

        x = next_x

    # Save Results
    results_dir = 'python_differential_mpc/results'
    os.makedirs(results_dir, exist_ok=True)

    # Convert lists to arrays
    log_t = np.array(log_t)
    log_x = np.array(log_x)
    log_u = np.array(log_u)
    log_compute_time = np.array(log_compute_time)

    np.savez(os.path.join(results_dir, 'logs.npz'), t=log_t, x=log_x, u=log_u, comp_time=log_compute_time)
    print("Simulation Complete. Results saved.")

if __name__ == "__main__":
    main()
