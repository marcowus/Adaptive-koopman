import numpy as np
import pandas as pd
from msd import MassSpringDamper
from scipy.signal import max_len_seq
import os

def generate_prbs(length, amplitude=1.0):
    # Estimate nbits needed
    nbits = int(np.ceil(np.log2(length + 1)))
    if nbits < 4: nbits = 4

    # Generate a sequence
    # Note: max_len_seq generates 2^nbits - 1 length
    seq = max_len_seq(nbits)[0]

    # Repeat if necessary
    while len(seq) < length:
        seq = np.concatenate((seq, max_len_seq(nbits)[0]))

    seq = seq[:length]
    # Map 0/1 to -amplitude/amplitude
    # max_len_seq usually returns 0s and 1s (or -1s, check doc)
    # Actually it returns 0 and 1.
    prbs = np.where(seq==1, amplitude, -amplitude)
    return prbs

def main():
    dt = 0.01
    sim_time = 100.0 # seconds, more data is better for EDMD
    steps = int(sim_time / dt)

    # Create system
    msd = MassSpringDamper(dt=dt)

    # Generate PRBS Input
    # Use randomized amplitude changes or just plain PRBS
    u_traj = generate_prbs(steps, amplitude=2.0)

    # Simulate
    x_traj = np.zeros((steps + 1, msd.state_dim))
    # Random initial condition
    x0 = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
    x_traj[0] = x0

    for k in range(steps):
        x_traj[k+1] = msd.step(x_traj[k], [u_traj[k]])

    # Save to CSV
    os.makedirs('python_differential_mpc/data', exist_ok=True)
    pd.DataFrame(x_traj).to_csv('python_differential_mpc/data/X.csv', index=False, header=False)
    pd.DataFrame(u_traj).to_csv('python_differential_mpc/data/U.csv', index=False, header=False)

    print(f"Generated data with {steps} steps.")

if __name__ == "__main__":
    main()
