from utils.cartpole_env import CartPoleBulletEnv
from utils.controllers import SimplePIDController
import matplotlib.pyplot as plt
import numpy as np

env = CartPoleBulletEnv(cart_mass=1.0, pendulum_mass=0.1, pendulum_len=0.5, render=True)
controller = SimplePIDController()
obs, _ = env.reset()

cart_vx_data = []
cart_acc_data = []
theta_data = []
theta_dot_data = []
pend_acc_data = []
time_steps = []

for t in range(1000):  # Simulating for 1000 steps
    action = controller.get_action(obs)
    obs, _, done, _, _ = env.step(action)

    cart_vx_data.append(obs[0])
    cart_acc_data.append(obs[1])
    theta_data.append(np.rad2deg(obs[2]))
    theta_dot_data.append(obs[3])
    pend_acc_data.append(obs[4])
    time_steps.append(t)

    # if done:
    #     break


env.close()

# Plot collected data
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(time_steps, cart_vx_data, label="Cart Velocity")
plt.xlabel("Time Steps")
plt.ylabel("Velocity")
plt.title("Cart Velocity over Time")
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(time_steps, cart_acc_data, label="Cart Acceleration", color="orange")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration")
plt.title("Cart Acceleration over Time")
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(time_steps, theta_data, label="Pendulum Angle", color="green")
plt.xlabel("Time Steps")
plt.ylabel("Angle (radians)")
plt.title("Pendulum Angle over Time")
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(time_steps, theta_dot_data, label="Pendulum Angular Velocity", color="red")
plt.xlabel("Time Steps")
plt.ylabel("Angular Velocity")
plt.title("Pendulum Angular Velocity over Time")
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(time_steps, pend_acc_data, label="Pendulum Acceleration", color="purple")
plt.xlabel("Time Steps")
plt.ylabel("Acceleration")
plt.title("Pendulum Acceleration over Time")
plt.legend()

plt.tight_layout()
plt.show()
