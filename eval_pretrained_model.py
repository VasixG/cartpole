import numpy as np, torch, pybullet as p
from utils.cartpole_env import CartPoleBulletEnv  # патч‑версия
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

device = "cuda" if torch.cuda.is_available() else "cpu"


class CartPole:
    def init(self):
        self.nx = 4
        self.nu = 1
        self.M = 1.0
        self.m = 0.1
        self.l = 0.5
        self.g = 9.81

    def f_torch(self, x, u):
        x_pos, x_dot, theta, theta_dot = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        F = u[:, 0]
        s, c = torch.sin(theta), torch.cos(theta)
        total_mass = self.M + self.m
        temp = (F + self.m * self.l * theta_dot**2 * s) / total_mass
        theta_ddot_num = self.g * s - c * temp
        theta_ddot_den = self.l * (4.0 / 3.0 - self.m * c**2 / total_mass)
        theta_ddot = theta_ddot_num / theta_ddot_den
        x_ddot = temp - (self.m * self.l * theta_ddot * c) / total_mass
        return torch.stack([x_dot, x_ddot, theta_dot, theta_ddot], dim=1)


class EllipsoidRegion:
    def init(self, axes=[0.5, 1.0, 0.1, 0.5], tol=1e-3):
        self.axes = np.array(axes)
        self.tol = tol

    def inside(self, x):
        return np.sum((x / self.axes) ** 2) < 1

    def boundary(self, x):
        val = np.sum((x / self.axes) ** 2)
        return abs(val - 1) < self.tol

    def sample_inside(self, N):
        result = []
        while len(result) < N:
            candidate = np.random.uniform(-self.axes, self.axes, size=(4,))
            if self.inside(candidate):
                result.append(candidate)
        return np.array(result, dtype=np.float32)

    def sample_on_boundary(self, N, scale=1.0):
        points = np.random.randn(N, 4)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        return (points * self.axes * scale).astype(np.float32)

    def sample_outside(self, N, scale=1.3):
        result = []
        box_min = -self.axes * scale
        box_max = self.axes * scale
        while len(result) < N:
            candidate = np.random.uniform(box_min, box_max, size=(4,))
            if not self.inside(candidate):
                result.append(candidate)
        return np.array(result, dtype=np.float32)

    def get_alpha(self, x):
        d = torch.sqrt(
            torch.sum((x / torch.tensor(self.axes, device=x.device)) ** 2, dim=1, keepdim=True)
        )
        alpha = 1 - d
        return torch.clamp(alpha, 0.0, 1.0)


class GeometryAwareW(nn.Module):
    def init(self, dims, regionA):
        super().init()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        self.regionA = regionA

    def forward(self, x):
        f = self.model(x)
        x0 = torch.zeros_like(x)
        f0 = self.model(x0)
        f = f - f0
        alpha = self.regionA.get_alpha(x)
        W = alpha * f + (1 - alpha) * 1.0
        return nn.ReLU()(W)


class ReluPositiveNetwork(nn.Module):
    def init(self, dims):
        super().init()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return 2 * (self.model(x))


class ReluNetwork(nn.Module):
    def init(self, dims):
        super().init()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def rk4_step(f, x, u, dt=0.01):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_trajectories(system, policy, initial_states, dt=0.01, num_steps=100):
    batch_size = initial_states.shape[0]
    states = [initial_states]
    x = initial_states
    for _ in range(num_steps):
        u = policy(x)
        x = rk4_step(system.f_torch, x, u, dt)
        states.append(x)
    return torch.stack(states, dim=0)


device = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.load(r"models_pretrained\best_policy8.pth", map_location=device)
policy.eval()

env = CartPoleBulletEnv(cart_mass=1.0, pendulum_mass=0.1, pendulum_len=0.5, render=True)


def make_state(obs):
    """
    Переводит bullet‑наблюдение в (x, x_dot, theta, theta_dot), ожидаемый сетью.
    x берём напрямую у базы тележки.
    """
    cart_pos, _ = p.getBasePositionAndOrientation(env._cart_id)
    x = cart_pos[0]  # положение вдоль X
    x_dot = obs[0]  # из obs = (x', x'', θ, θ', θ'')
    theta, theta_dot = obs[2], obs[3]
    return torch.tensor([[x, x_dot, theta, theta_dot]], dtype=torch.float32, device=device)


for ep in range(5):
    obs = env.reset()
    done = truncated = False
    while not (done or truncated):
        state = obs
        with torch.no_grad():
            F = policy(torch.Tensor(state).to(device)).item()  # сила (Н)
        action = np.array([30 * F], dtype=np.float32)
        obs, _, done, truncated, _ = env.step(action)
env.close()
