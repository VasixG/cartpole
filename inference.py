import time
import numpy as np
from utils.cartpole_env import CartPoleBulletEnv
from stable_baselines3 import PPO


def main():
    env = CartPoleBulletEnv(cart_mass=0.2, pendulum_len=0.3, pendulum_mass=0.2, render=True)
    env.reset()

    # Load the trained PPO model
    model = PPO.load("models/ppo_cartpole_bullet.zip", env=env)

    obs, _ = env.reset()  # Extract the observation from the tuple
    done = False
    while True:
        action, _states = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
