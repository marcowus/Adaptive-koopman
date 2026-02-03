import numpy as np
import pandas as pd
from scipy.linalg import pinv

class KoopmanEDMD:
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.lift_dim = None

    def lift(self, x):
        # x shape: (N, 2) or (2,)
        original_ndim = x.ndim
        if original_ndim == 1:
            x = x.reshape(1, -1)

        x1 = x[:, 0]
        x2 = x[:, 1]

        # Polynomials up to degree 2
        # [x1, x2, x1^2, x1*x2, x2^2, 1]

        z = np.stack([
            x1,
            x2,
            x1**2,
            x1*x2,
            x2**2,
            np.ones_like(x1)
        ], axis=1)

        if original_ndim == 1:
            return z.flatten()
        return z

    def fit(self, X_csv, U_csv):
        X_data = pd.read_csv(X_csv, header=None).values
        U_data = pd.read_csv(U_csv, header=None).values

        # Prepare data
        # X has N+1 steps, U has N steps
        # Ensure lengths match
        min_len = min(X_data.shape[0]-1, U_data.shape[0])

        X_k = X_data[:min_len]
        X_kp1 = X_data[1:min_len+1]
        U_k = U_data[:min_len]

        # Lift
        Z_k = self.lift(X_k) # (N, lift_dim)
        Z_kp1 = self.lift(X_kp1)

        self.lift_dim = Z_k.shape[1]

        # Build regression matrices
        # We want Z_{k+1} = A * Z_k + B * U_k
        # Transpose to column vectors layout: Z = [z0, z1, ...]
        Z = Z_k.T
        Z_prime = Z_kp1.T
        U = U_k.T

        # Regressors: [Z; U]
        Omega = np.vstack([Z, U])

        # Solve for G = [A, B] such that Z_prime = G * Omega
        # G = Z_prime * pinv(Omega)
        G = Z_prime @ pinv(Omega)

        n_z = Z.shape[0]
        self.A = G[:, :n_z]
        self.B = G[:, n_z:]

        # Solve for C: X_k = C * Z_k
        # C = X_k.T * pinv(Z_k.T)
        self.C = X_k.T @ pinv(Z)

        print(f"Koopman fitted. A shape: {self.A.shape}, B shape: {self.B.shape}, C shape: {self.C.shape}")

    def predict(self, z, u):
        # z: (lift_dim,)
        # u: (control_dim,) or scalar
        if np.isscalar(u):
            u = np.array([u])

        z_next = self.A @ z + self.B @ u
        return z_next

    def get_matrices(self):
        return self.A, self.B, self.C
