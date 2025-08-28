"""Demonstration of fault-tolerant control using Q-statistic detection and MPC."""
import numpy as np
from fault_detection import QStatFaultDetector


def lqr_gain(A, B, Q, R, N):
    P = Q.copy()
    for _ in range(N-1):
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P = Q + A.T @ P @ (A - B @ K)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K

A_nominal = np.array([[1.0]])
B_nominal = np.array([[0.1]])
A_fault = np.array([[1.2]])
B_fault = np.array([[0.5]])

Q = np.array([[1.0]])
R = np.array([[0.1]])
N = 10

fd = QStatFaultDetector(residual_dim=1, alpha=0.99, window=20)

x = np.array([[1.0]])
A_est = A_nominal.copy()
B_est = B_nominal.copy()

fault_time = 20
collect = False
Phi = []
Y = []

for k in range(60):
    K = lqr_gain(A_est, B_est, Q, R, N)
    u = -K @ x
    if k < fault_time:
        A_true, B_true = A_nominal, B_nominal
    else:
        A_true, B_true = A_fault, B_fault
    x_true = A_true @ x + B_true @ u
    x_pred = A_est @ x + B_est @ u
    residual = x_true - x_pred
    fault, q_val = fd.update(residual.ravel())
    if fault and not collect:
        print(f"Fault detected at step {k}, Q={q_val:.3f}")
        collect = True
    if collect:
        Phi.append([x[0,0], u[0,0]])
        Y.append(x_true[0,0])
        if len(Phi) >= 2:
            theta, *_ = np.linalg.lstsq(np.array(Phi), np.array(Y), rcond=None)
            A_est = np.array([[theta[0]]])
            B_est = np.array([[theta[1]]])
            collect = False
    x = x_true

print("Final estimated A,B:", A_est[0,0], B_est[0,0])
print("Final state:", x[0,0])
