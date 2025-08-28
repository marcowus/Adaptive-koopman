"""Demonstration of Q-statistic fault detection with a simple adaptive controller."""
import numpy as np
from core.adapt_net_linear import AdaptNet_linear
from fault_detection import QStatFaultDetector

# Simple 1D system x_{k+1} = A x_k + B u_k
A_nominal = 1.0
B_nominal = 0.1
A_fault = 1.2  # faulty dynamics after fault

# Adaptation network parameters (minimal)
adapt_params = {
    'state_dim': 1,
    'ctrl_dim': 1,
    'lift_dim': 1,
    'first_obs_const': 0,
    'override_C': False,
    'batch_size': 1,
    'epochs': 20,
    'optimizer': 'adam',
    'lr': 0.05,
    'l1_reg': 0.0,
    'l2_reg': 0.0,
    'warm_start': False
}

adapt_net = AdaptNet_linear(adapt_params)

# Fault detector on 1D residual
fd = QStatFaultDetector(residual_dim=1, alpha=0.99, window=20)

# initial state
x = np.array([1.0])
A_est = A_nominal
B_est = B_nominal

for k in range(100):
    u = -0.2 * x  # simple stabilizing control
    x_pred = A_est * x + B_est * u
    A_true = A_nominal if k < 60 else A_fault
    x_true = A_true * x + B_nominal * u
    residual = x_true - x_pred
    fault, q_val = fd.update(residual)
    if fault:
        print(f"Fault detected at step {k}, Q={q_val:.3f}")
        # Prepare data for adaptation network
        Z = np.array([[x[0]]])
        U = np.array([[u[0]]])
        del_Z = np.array([[residual[0]]])
        adapt_net.model_pipeline(Z, U, del_Z, print_epoch=False)
        dA, dB = adapt_net.get_del_matrices()
        A_est += dA[0, 0]
        B_est += dB[0, 0]
    x = x_true

print("Final estimated A:", A_est)
