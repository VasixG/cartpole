import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def make_ppo_model(vec_env: gym.Env, **kwargs):

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        # batch_size=64,
        learning_rate=1e-3,
        verbose=1,
        device="cpu",
    )

    return model
