import numpy as np
import matplotlib.pyplot as plt
import os

def visualize():
    results_path = 'python_differential_mpc/results/logs.npz'
    if not os.path.exists(results_path):
        print("Results not found.")
        return

    data = np.load(results_path)
    t = data['t']
    x = data['x']
    u = data['u']
    comp_time = data['comp_time']

    # Set style for academic papers
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    # 1. State Tracking
    plt.figure(figsize=(10, 6))
    plt.plot(t, x[:, 0], label='Position $p$ (m)', linewidth=2)
    plt.plot(t, x[:, 1], label='Velocity $v$ (m/s)', linewidth=2, linestyle='--')
    plt.axhline(0, color='black', linestyle=':', linewidth=1)
    # Add vertical line for disturbance
    plt.axvline(15.0, color='red', linestyle='-.', alpha=0.5, label='Disturbance')

    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.title('System State Response')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('python_differential_mpc/results/state_tracking.png', dpi=300)
    plt.close()

    # 2. Inputs
    plt.figure(figsize=(10, 4))
    plt.step(t, u, label='Control Input $u$ (N)', where='post', color='tab:red', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Control Input')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('python_differential_mpc/results/inputs.png', dpi=300)
    plt.close()

    # 3. Computation Time
    plt.figure(figsize=(10, 4))
    mask = comp_time > 1e-6 # Filter effectively zero values
    if np.any(mask):
        plt.plot(t[mask], comp_time[mask]*1000, 'o-', markersize=4, color='tab:green', label='Computation Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Time (ms)')
        plt.title('Optimization Computation Time (Log Scale)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, which="both")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('python_differential_mpc/results/computation_time.png', dpi=300)
    else:
        print("No computation time data found (or all zeros).")
    plt.close()

    print("Plots generated in python_differential_mpc/results/")

if __name__ == "__main__":
    visualize()
