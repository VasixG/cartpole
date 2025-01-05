from utils.cartpole_env import CartPoleBulletEnv
from stable_baselines3.common.env_util import make_vec_env
from utils.models import make_ppo_model


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

    n_envs = 4

    vec_env = make_vec_env(
        make_env(cart_mass=30, pendulum_len=0.6, pendulum_mass=0.5, render=False), n_envs=n_envs
    )

    model = make_ppo_model(vec_env)

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    model.save("models/ppo_cartpole_bullet2.zip")

    vec_env.close()


if __name__ == "__main__":
    main()
