from utils.cartpole_env import CartPoleBulletEnv
from stable_baselines3.common.env_util import make_vec_env
from utils.models import make_ppo_model
import os
from stable_baselines3 import PPO


def make_env(cart_mass, pendulum_len, pendulum_mass, render=False):
    def _init():
        return CartPoleBulletEnv(
            cart_mass=cart_mass,
            pendulum_len=pendulum_len,
            pendulum_mass=pendulum_mass,
            render=render,
        )

    return _init


def main():

    n_envs = 1

    vec_env = make_vec_env(
        make_env(cart_mass=1.1, pendulum_len=0.6, pendulum_mass=0.1, render=True), n_envs=n_envs
    )

    model_path = "models/ppo_cartpole_bullet2.zip"

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("No pre-trained model found. Creating a new model...")
        model = make_ppo_model(vec_env)

    # model = make_ppo_model(vec_env)

    model_path = "models/ppo_cartpole_bullet2.zip"

    if not os.path.exists("models"):
        os.makedirs("models")

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("No pre-trained model found. Creating a new model...")
        model = make_ppo_model(vec_env)

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    model.save(model_path)

    if vec_env is not None:
        vec_env.close()


if __name__ == "__main__":
    main()
