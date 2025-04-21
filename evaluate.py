from utils.cartpole_env import CartPoleBulletEnv
from stable_baselines3.common.env_util import make_vec_env
from utils.models import make_ppo_model
import os
from stable_baselines3 import PPO
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import time

device = "cuda" if torch.cuda.is_available() else "cpu"


class CartPole:
    def __init__(self):
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
    def __init__(self, axes=[0.5, 1.0, 0.1, 0.5], tol=1e-3):
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
    def __init__(self, dims, regionA):
        super().__init__()
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
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return 20 * self.model(x)


class ReluNetwork(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return 100 * self.model(x)


def rk4_step(f, x, u, dt=0.01):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def make_env(cart_mass, pendulum_len, pendulum_mass, render=False):
    def _init():
        return CartPoleBulletEnv(
            cart_mass=cart_mass,
            pendulum_len=pendulum_len,
            pendulum_mass=pendulum_mass,
            render=render,
        )

    return _init


import threading
from queue import Queue


def simulate_trajectories_pybullet(env, policy, initial_states, num_steps=100):
    """
    Simulates trajectories in parallel using multithreading for multiple initial states with a single environment.

    Args:
        env: Single environment instance (e.g., CartPoleBulletEnv).
        policy: Trained policy (e.g., neural network) to generate actions.
        initial_states: Array of initial states, shape [n_initial_states, state_dim].
        num_steps: Number of steps to simulate per trajectory (default: 100).

    Returns:
        torch.Tensor: Trajectories, shape [num_steps + 1, n_initial_states, state_dim].
    """
    n_initial_states = len(initial_states)
    trajectories = [None] * n_initial_states  # Placeholder for each trajectory
    queue = Queue()  # Queue to collect results from threads
    lock = threading.Lock()  # Lock to ensure thread-safe environment access

    def simulate_single_trajectory(state, index):
        """
        Simulates a single trajectory starting from a given initial state.

        Args:
            state: Initial state for the trajectory.
            index: Index to store the result in trajectories list.
        """
        # Create a local trajectory list
        traj = []
        done = False

        # Thread-safe environment reset
        with lock:
            env.set_initial_state(state)
            obs = env.reset()

        traj.append(obs.copy())  # Record initial observation

        # Simulate for num_steps
        for _ in range(num_steps):
            if not done:
                # Compute action using the policy
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().to(device)
                    action = policy(obs_t).cpu().numpy()
                # Thread-safe environment step
                with lock:
                    obs, reward, done, _, info = env.step(action)
            # Append current observation (repeat last obs if done)
            traj.append(obs.copy())

        # Stack trajectory into a numpy array: [num_steps + 1, state_dim]
        traj_array = np.stack(traj, axis=0)
        queue.put((index, traj_array))

    # Create and start a thread for each initial state
    threads = []
    for i, state in enumerate(initial_states):
        thread = threading.Thread(target=simulate_single_trajectory, args=(state, i))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Collect results from the queue
    while not queue.empty():
        index, traj_array = queue.get()
        trajectories[index] = traj_array

    # Stack all trajectories: [num_steps + 1, n_initial_states, state_dim]
    states = np.stack(trajectories, axis=1)

    # Convert to tensor and move to device
    return torch.from_numpy(states).float().to(device)


def evaluate_policy(vec_env, policy, theta_threshold=np.pi / 12, max_steps=1000):
    """
    Simulate the cartpole and return the number of steps the pole stays in the 'up' position.
    Args:
        vec_env: Vectorized cartpole environment
        policy: Current policy (PyTorch model)
        theta_threshold: Maximum angle (in radians) for the pole to be considered 'up'
        max_steps: Maximum steps to simulate
    Returns:
        steps_up: Number of steps the pole stayed up
    """
    obs = vec_env.reset()
    steps_up = 0
    for step in range(max_steps):
        # Get action from the policy
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            action = policy(obs_t).cpu().numpy()
        # Step the environment
        obs, _, dones, _, _ = vec_env.step(action)
        # Check the pole's angle (assuming theta is the 3rd element in obs)
        theta = obs[2]  # Adjust index based on your environment's observation space
        if np.all(np.abs(theta) < theta_threshold):
            steps_up += 1
        else:
            break  # Pole has fallen
        if np.any(dones):
            break  # Episode ended
    return steps_up


def render_trained_policy(vec_env, policy, num_steps=1000):
    """
    Render the cartpole system using the trained policy.
    Args:
        vec_env: Vectorized cartpole environment with rendering enabled
        policy: Trained policy (PyTorch model)
        num_steps: Number of steps to render
    """
    obs = vec_env.reset()
    for step in range(num_steps):
        # Get action from the trained policy
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            action = policy(obs_t).cpu().numpy()
        # Step the environment and render
        obs, _, dones, _ = vec_env.step(action)
        vec_env.render()  # Display the current frame
        if np.any(dones):
            break  # Episode ended


def render_trained_policy(vec_env, policy, num_steps=1000, sleep_time=0.01):
    """
    Render the cartpole system using the trained policy with a delay between steps.

    Args:
        vec_env: Vectorized cartpole environment with rendering enabled
        policy: Trained policy (PyTorch model)
        num_steps: Number of steps to render (default: 1000)
        sleep_time: Time to sleep between steps in seconds (default: 0.01)
    """
    obs = vec_env.reset()
    for step in range(num_steps):
        # Get action from the trained policy
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().to(device)
            action = policy(obs_t).cpu().numpy()
            print(action)
        # Step the environment and render
        obs, _, dones, _, _ = vec_env.step(action)
        vec_env.render()  # Display the current frame
        time.sleep(sleep_time)  # Pause to make rendering observable
        if np.any(dones):
            break  # Episode ended


def main():

    vec_env = CartPoleBulletEnv(
        cart_mass=1.0,
        pendulum_len=0.6,
        pendulum_mass=0.1,
        render="human",
    )

    policy_path = r"C:\Users\vasil\study\vkr\cartpole\models\20250320_233857_950_80.00\policy.pth"

    policy = ReluNetwork(dims=[4, 64, 64, 32, 1]).to(device)

    policy.load_state_dict(torch.load(policy_path, map_location=device))

    policy.eval().to(device)

    render_trained_policy(vec_env, policy)


if __name__ == "__main__":
    main()
