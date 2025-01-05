import pybullet as p
import pybullet_data
import time

#############################################
# 1) ПАРАМЕТРЫ ТЕЛЕЖКИ, КОЛЁС И МАЯТНИКА
#############################################

# Тележка
cart_halfExtents = [0.2, 0.1, 0.05]
# Итого размеры: длина=0.4 (X), ширина=0.2 (Y), высота=0.1 (Z).

# Ставим тележку так, чтобы её нижняя грань была на Z=0:
cart_start_position = [0, 0, cart_halfExtents[2]]  # центр тележки на Z=0.05
cart_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

# Колёса
wheel_radius = 0.05
wheel_width = 0.02
wheel_mass = 0.2

wheel_center_z = -cart_halfExtents[2] - wheel_radius / 2
wheel_positions = [
    [+cart_halfExtents[0] * 3 / 5, +cart_halfExtents[1], wheel_center_z],
    [+cart_halfExtents[0] * 3 / 5, -cart_halfExtents[1], wheel_center_z],
    [-cart_halfExtents[0] * 3 / 5, +cart_halfExtents[1], wheel_center_z],
    [-cart_halfExtents[0] * 3 / 5, -cart_halfExtents[1], wheel_center_z],
]
wheel_orient = p.getQuaternionFromEuler([1.57, 0, 0])  # колёса вдоль Y
wheel_orientations = [wheel_orient] * 4
wheel_joint_axis = [0, 0, 1]

# Маятник
pendulum_length = 3
pendulum_radius = 0.01
pendulum_mass = 1

# Шарнир хотим на верхней грани тележки (X=0, Y=0, Z=+cart_halfExtents[2]):
pendulum_position = [0, 0, +cart_halfExtents[2]]
# Ось вращения маятника: вокруг Y, чтобы качался в плоскости XZ:
pendulum_joint_axis = [0, 1, 0]

#############################################
# 2) ИНИЦИАЛИЗАЦИЯ PYBULLET
#############################################
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

plane_id = p.loadURDF("plane.urdf")

#############################################
# 3) COLLISION / VISUAL SHAPES
#############################################

# Тележка
cart_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cart_halfExtents)
cart_visual_shape = p.createVisualShape(
    p.GEOM_BOX, halfExtents=cart_halfExtents, rgbaColor=[0.2, 0.2, 0.8, 1]
)

# Колёса
wheel_collision_shape = p.createCollisionShape(
    p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_width
)
wheel_visual_shape = p.createVisualShape(
    p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width, rgbaColor=[0, 0.8, 0.2, 1]
)

# Маятник:
# Чтобы нижняя часть цилиндра совпадала с pivot, укажем:
#   collisionFramePosition=[0,0,+pendulum_length/2],
#   visualFramePosition=[0,0,+pendulum_length/2].
# Тогда цилиндр будет рисоваться (и коллизиться) от z=0 до z=+L,
# где z=0 - это pivot (точка шарнира).
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

#############################################
# 4) МАССИВЫ ДЛЯ MULTIBODY (4 КОЛЕСА + 1 МАЯТНИК)
#############################################
linkMasses = [wheel_mass] * 4 + [pendulum_mass]
linkCollisionShapeIndices = [
    wheel_collision_shape,
    wheel_collision_shape,
    wheel_collision_shape,
    wheel_collision_shape,
    pendulum_collision_shape,
]
linkVisualShapeIndices = [
    wheel_visual_shape,
    wheel_visual_shape,
    wheel_visual_shape,
    wheel_visual_shape,
    pendulum_visual_shape,
]

linkPositions = wheel_positions + [pendulum_position]
linkOrientations = wheel_orientations + [[0, 0, 0, 1]]
linkParentIndices = [0, 0, 0, 0, 0]
linkJointTypes = [
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
    p.JOINT_REVOLUTE,
]
linkJointAxis = [
    wheel_joint_axis,
    wheel_joint_axis,
    wheel_joint_axis,
    wheel_joint_axis,
    pendulum_joint_axis,
]

# Важно: инерциальная рамка (центр масс) маятника
# тоже должна быть на середине стержня (т.е. +L/2 от pivot'а).
linkInertialFramePositions = [[0, 0, 0]] * 4 + [[0, 0, +pendulum_length / 2]]
linkInertialFrameOrientations = [[0, 0, 0, 1]] * 4 + [[0, 0, 0, 1]]

#############################################
# 5) СОЗДАЁМ MULTIBODY (ТЕЛЕЖКА + КОЛЁСА + МАЯТНИК)
#############################################
cart_id = p.createMultiBody(
    baseMass=1.0,
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


#############################################
# 6) УПРАВЛЕНИЕ
#############################################
def control_wheels(velocity):
    for j in range(4):
        p.setJointMotorControl2(
            bodyUniqueId=cart_id,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=velocity,
            force=10,
        )


# Маятник (звено 4): отключаем мотор, чтобы он мог качаться
p.setJointMotorControl2(
    bodyUniqueId=cart_id, jointIndex=4, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0
)

# Зададим небольшой начальный угол (0.3 рад)
p.resetJointState(cart_id, jointIndex=4, targetValue=0.3)

#############################################
# 7) КАМЕРА
#############################################
p.resetDebugVisualizerCamera(
    cameraDistance=2.0, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2]
)

#############################################
# 8) ШАГИ СИМУЛЯЦИИ
#############################################
for step in range(10000):
    control_wheels(10)
    p.stepSimulation()
    time.sleep(1 / 240)

p.disconnect()
