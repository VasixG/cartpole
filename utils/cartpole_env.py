import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import gymnasium as gym
from gymnasium import spaces


class CartPoleBulletEnv(gym.Env):
    """
    PyBullet-среда для тележки с перевернутым маятником.

    Наблюдения (состояние), которое мы возвращаем (пример):
    1) x'  -- скорость тележки
    2) x'' -- ускорение тележки (приближённо считаем разницу скоростей)
    3) theta -- угол маятника (от вертикали, радианы)
    4) theta' -- угловая скорость
    5) theta'' -- угловое ускорение (приближённо)

    Действие: сила (или целевая скорость колёс).
    """

    def __init__(self, cart_mass, pendulum_mass, pendulum_len, render=False):
        super(CartPoleBulletEnv, self).__init__()
        self.client_id = p.connect(p.GUI if render else p.DIRECT)

        # Инициализация гиперпараметров
        self.cart_mass = cart_mass
        self.pendulum_mass = pendulum_mass
        self.pendulum_len = pendulum_len

        # Пределы по силе, которую можем подать
        self.action_space = spaces.Box(
            low=np.array([-100.0], dtype=np.float32),
            high=np.array([+100.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Наблюдения: (x', x'', theta, theta', theta'')
        high = np.array([100] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render
        self._time_step = 1.0 / 120.0  # шаг симуляции
        self._max_episode_steps = 2048
        self._elapsed_steps = 0

        # Для вычисления ускорений будем хранить предыдущие значения скоростей
        self._prev_cart_vel = 0.0
        self._prev_pend_vel = 0.0

        # IDs которые появятся после reset()
        self._plane_id = None
        self._cart_id = None

    def _build_scene(self):
        """Создаём плоскость и тележку с маятником (MultiBody)."""
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Ускоряем симуляцию
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=30,  # Оптимальный баланс
            enableConeFriction=1,
        )

        # Плоскость
        self._plane_id = p.loadURDF("plane.urdf")

        #########################################################
        # Ниже — создание тележки + 4 колеса + маятник
        #########################################################
        cart_halfExtents = [0.2, 0.1, 0.05]
        cart_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        wheel_radius = 0.05
        wheel_width = 0.02
        wheel_mass = 0.5
        wheel_center_z = -cart_halfExtents[2] - wheel_radius / 2

        wheel_positions = [
            [+cart_halfExtents[0] * 3 / 5, +cart_halfExtents[1], wheel_center_z],
            [+cart_halfExtents[0] * 3 / 5, -cart_halfExtents[1], wheel_center_z],
            [-cart_halfExtents[0] * 3 / 5, +cart_halfExtents[1], wheel_center_z],
            [-cart_halfExtents[0] * 3 / 5, -cart_halfExtents[1], wheel_center_z],
        ]

        cart_start_position = [0, 0, cart_halfExtents[2] + 3 / 2 * wheel_radius]

        wheel_orient = p.getQuaternionFromEuler([1.57, 0, 0])
        wheel_orientations = [wheel_orient] * 4
        wheel_joint_axis = [0, 0, 1]

        pendulum_length = self.pendulum_len
        pendulum_radius = 0.01
        pendulum_mass = self.pendulum_mass

        pendulum_position = [0, 0, +cart_halfExtents[2]]
        pendulum_joint_axis = [0, 1, 0]

        # Создаём collision/visual
        cart_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cart_halfExtents)
        cart_visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=cart_halfExtents, rgbaColor=[0.2, 0.2, 0.8, 1]
        )

        wheel_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_width
        )
        wheel_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width, rgbaColor=[0, 0.8, 0.2, 1]
        )

        # Маятник: сдвигаем визуал/коллизию вверх на +L/2, чтобы pivot = нижняя точка
        pendulum_collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=pendulum_radius,
            height=pendulum_length,
            collisionFramePosition=[0, 0, +pendulum_length / 2],
        )
        pendulum_visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=pendulum_radius,
            length=pendulum_length,
            rgbaColor=[1, 0, 0, 1],
            visualFramePosition=[0, 0, +pendulum_length / 2],
        )

        linkMasses = [wheel_mass] * 4 + [pendulum_mass]
        linkCollisionShapeIndices = [wheel_collision_shape] * 4 + [pendulum_collision_shape]
        linkVisualShapeIndices = [wheel_visual_shape] * 4 + [pendulum_visual_shape]
        linkPositions = wheel_positions + [pendulum_position]
        linkOrientations = wheel_orientations + [[0, 0, 0, 1]]
        linkParentIndices = [0, 0, 0, 0, 0]
        linkJointTypes = [p.JOINT_REVOLUTE] * 4 + [p.JOINT_REVOLUTE]
        linkJointAxis = [wheel_joint_axis] * 4 + [pendulum_joint_axis]
        linkInertialFramePositions = [[0, 0, 0]] * 4 + [[0, 0, pendulum_length / 2]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]] * 4 + [[0, 0, 0, 1]]

        self._cart_id = p.createMultiBody(
            baseMass=self.cart_mass,
            baseCollisionShapeIndex=cart_collision_shape,
            baseVisualShapeIndex=cart_visual_shape,
            basePosition=cart_start_position,
            baseOrientation=cart_start_orientation,
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
        )

        # Отключаем мотор маятника (чтобы свободно качался)
        # Индекс маятника = 4
        p.resetJointState(self._cart_id, jointIndex=4, targetValue=0)
        p.setJointMotorControl2(
            bodyUniqueId=self._cart_id,
            jointIndex=4,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0,
        )

        if self.render_mode:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2]
            )

    def reset(self, seed=None, options=None):
        """Сброс среды: начинаем новый эпизод."""
        super().reset(seed=seed)  # If using seeding
        self._build_scene()
        self._elapsed_steps = 0

        # Зададим маятнику небольшой стартовый угол
        init_angle = np.random.uniform(-0.05, 0.05)
        p.resetJointState(self._cart_id, jointIndex=4, targetValue=init_angle)

        self._prev_cart_vel = 0.0
        self._prev_pend_vel = 0.0

        obs = self._get_observation()
        info = {}  # Add additional metadata if needed

        return obs, info

    def step(self, action):
        """Применяем действие (скорость колёс) и шагаем в PyBullet."""
        velocity = action[0]
        for j in range(4):
            p.setJointMotorControl2(
                bodyUniqueId=self._cart_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=velocity,
                force=10,
            )

        p.stepSimulation()
        # time.sleep(self._time_step)
        self._elapsed_steps += 1

        # Получение текущей позиции тележки
        cart_position, _ = p.getBasePositionAndOrientation(self._cart_id)
        cart_x, cart_y, cart_z = cart_position

        # Ограничения по положениям
        x_limit = 3.0
        y_limit = 3.0
        out_of_bounds = abs(cart_x) > x_limit or abs(cart_y) > y_limit

        # Получение состояния
        obs = self._get_observation()
        cart_vx = obs[0]  # скорость тележки
        theta = obs[2]  # угол маятника
        theta_dot = obs[3]  # угловая скорость маятника

        # Улучшенное вознаграждение
        reward = 1.0 - abs(theta) * 0.05  # основной бонус за вертикальность маятника
        reward -= abs(cart_vx) * 0.02  # штраф за большую скорость тележки
        reward -= abs(cart_x) * 0.03  # штраф за уход тележки от центра
        reward -= abs(theta_dot) * 0.005  # штраф за колебания маятника

        # Условия завершения эпизода
        done = abs(theta) > math.pi / 4 or out_of_bounds
        truncated = self._elapsed_steps >= self._max_episode_steps

        info = {"out_of_bounds": out_of_bounds}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        """
        Формируем вектор (x', x'', theta, theta', theta'').
        """
        # Скорость тележки: это линейная скорость базы
        vel_cart, _ = p.getBaseVelocity(self._cart_id)
        cart_vx = vel_cart[0]  # допустим, вдоль X

        # Маятник: jointIndex=4
        # getJointState -> (pos, vel, reactionForces, torque)
        pend_state = p.getJointState(self._cart_id, 4)
        theta = pend_state[0]  # (радианы) pos
        theta_dot = pend_state[1]  # угловая скорость

        # Аппроксимация ускорения как delta(vel)/delta_t
        cart_acc = (cart_vx - self._prev_cart_vel) / self._time_step
        pend_acc = (theta_dot - self._prev_pend_vel) / self._time_step

        self._prev_cart_vel = cart_vx
        self._prev_pend_vel = theta_dot

        return np.array([cart_vx, cart_acc, theta, theta_dot, pend_acc], dtype=np.float32)

    def render(self, mode="human"):
        """Если нужно, включите p.connect(p.GUI) в __init__."""
        pass

    def close(self):
        if p.getConnectionInfo()["isConnected"]:
            p.disconnect()
