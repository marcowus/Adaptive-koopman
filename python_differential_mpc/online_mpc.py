import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DifferentialKoopmanMPC(nn.Module):
    def __init__(self, A0, B0, C0, horizon=50, dt=0.01):
        super().__init__()
        self.dt = dt
        self.horizon = horizon

        # Convert initial matrices to buffers (fixed base)
        self.register_buffer('A0', torch.FloatTensor(A0))
        self.register_buffer('B0', torch.FloatTensor(B0))
        self.register_buffer('C0', torch.FloatTensor(C0))

        self.state_dim = C0.shape[0]
        self.lift_dim = A0.shape[0]
        self.ctrl_dim = B0.shape[1]

        # Trainable corrections
        self.dA = nn.Parameter(torch.zeros_like(self.A0))
        self.dB = nn.Parameter(torch.zeros_like(self.B0))
        self.dC = nn.Parameter(torch.zeros_like(self.C0))

        # Trainable Plan U_mpc
        # Horizon H, control dim M
        # Initialize with zeros
        self.U_mpc = nn.Parameter(torch.zeros(horizon, self.ctrl_dim))

        # Weights
        self.w_m = 1.0 # Model weight
        self.w_p = 1.0 # MPC weight (tuned for stability)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.005)

    def forward(self, z_init):
        # Predict future trajectory using current model and U_mpc
        # z_init: (lift_dim,)

        A = self.A0 + self.dA
        B = self.B0 + self.dB
        C = self.C0 + self.dC

        z = z_init
        x_pred = []
        u_pred = []

        for k in range(self.horizon):
            u = self.U_mpc[k]
            # z_next = A z + B u
            z = torch.mv(A, z) + torch.mv(B, u)
            x = torch.mv(C, z)

            x_pred.append(x)
            u_pred.append(u)

        return torch.stack(x_pred), torch.stack(u_pred)

    def model_loss(self, z_history, u_history):
        # z_history: (H_past+1, lift_dim)
        # u_history: (H_past, ctrl_dim)

        if len(u_history) == 0:
            return torch.tensor(0.0)

        A = self.A0 + self.dA
        B = self.B0 + self.dB

        z_k = z_history[:-1]
        z_kp1_true = z_history[1:]
        u_k = u_history

        # Batch prediction
        # z_kp1_pred = (A @ z_k.T + B @ u_k.T).T

        z_kp1_pred = (torch.mm(A, z_k.T) + torch.mm(B, u_k.T)).T

        loss = torch.mean((z_kp1_pred - z_kp1_true)**2)

        return loss

    def mpc_cost(self, x_pred, u_pred, ref_x):
        # x_pred: (H, state_dim)
        # u_pred: (H, ctrl_dim)
        # ref_x: target state (0,0)

        # Q = diag([10, 5]) - Penalize position and velocity
        # R = diag([0.1])

        Q = torch.tensor([10.0, 5.0], device=x_pred.device)
        R = torch.tensor([0.1], device=x_pred.device)

        # Calculate cost
        x_err = x_pred - ref_x
        cost_x = torch.mean(torch.sum(x_err**2 * Q, dim=1))
        cost_u = torch.mean(torch.sum(u_pred**2 * R, dim=1))

        return cost_x + cost_u

    def optimize_step(self, history_z, history_u, current_z, steps=5):
        # history_z, history_u are lists or numpy arrays
        # current_z is numpy array

        # Convert to tensors
        hz = torch.FloatTensor(np.array(history_z))
        hu = torch.FloatTensor(np.array(history_u))
        cz = torch.FloatTensor(current_z)

        ref_x = torch.zeros(self.state_dim)

        for _ in range(steps):
            self.optimizer.zero_grad()

            # Model Loss
            l_model = self.model_loss(hz, hu)

            # MPC Rollout & Cost
            x_pred, u_pred = self(cz)
            l_mpc = self.mpc_cost(x_pred, u_pred, ref_x)

            # Joint Loss
            loss = self.w_m * l_model + self.w_p * l_mpc

            loss.backward()

            # Clip gradients if necessary
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

            self.optimizer.step()

        return loss.item(), l_model.item(), l_mpc.item()

    def get_control(self):
        return self.U_mpc[0].detach().numpy()

    def shift_plan(self, shift_steps=1):
        # Shift U_mpc for warm start
        with torch.no_grad():
            new_u = torch.zeros_like(self.U_mpc)
            if shift_steps < self.horizon:
                new_u[:-shift_steps] = self.U_mpc[shift_steps:]
                # Fill the end with the last value or zero
                new_u[-shift_steps:] = self.U_mpc[-1]
            else:
                # Reset if shift is larger than horizon
                pass
            self.U_mpc.copy_(new_u)
