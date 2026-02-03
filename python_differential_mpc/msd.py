import numpy as np

class MassSpringDamper:
    def __init__(self, m=1.0, k=1.0, c=0.5, dt=0.01):
        self.m = m
        self.k = k
        self.c = c
        self.dt = dt
        self.state_dim = 2
        self.control_dim = 1

    def dynamics(self, state, u):
        p, v = state
        # Ensure u is scalar or extract first element
        u_val = u[0] if hasattr(u, '__getitem__') else u

        dp = v
        dv = -(self.k/self.m)*p - (self.c/self.m)*v + (1.0/self.m)*u_val
        return np.array([dp, dv])

    def step(self, state, u):
        # RK4 integration
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)

        next_state = state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return next_state
