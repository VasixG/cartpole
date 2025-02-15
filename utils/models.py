import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def make_ppo_model(vec_env: gym.Env, **kwargs):

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=64,
        learning_rate=5e-5,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
        device="cpu",
    )

    return model
