import numpy as np


class RandomController:

    def __init__(self, action_low=-10.0, action_high=10.0):
        self.action_low = action_low
        self.action_high = action_high

    def get_action(self, obs):
        return np.array([np.random.uniform(self.action_low, self.action_high)])


class SimplePIDController:

    def __init__(self, kp=10.0, kd=5.0, max_force=10.0):
        self.kp = kp
        self.kd = kd
        self.max_force = max_force

    def get_action(self, obs):
        # obs = [cart_vx, cart_acc, theta, theta_dot, pend_acc]
        theta = obs[2]
        theta_dot = obs[3]
        force = -(self.kp * theta + self.kd * theta_dot)
        force = np.clip(force, -self.max_force, self.max_force)
        return np.array([force])


class RlPolicyController:

    def __init__(self, rl_model):

        self.model = rl_model

    def get_action(self, obs):
        action, _states = self.model.predict(obs, deterministic=True)
        return action
