import pybullet as p
import pybullet_data
import numpy as np
import time as t

# Variables
PATH = 'quadrupbot_env\\quadrupv1.urdf'
# PATH = 'quadrupbot_env\\quadrup\\urdf\\Model_Doan_HK231_ASM_Dog_URDF.urdf'
sleep_time = 1./240.
initial_height = 0.2937
initial_ori = [0,0,0,1]
jointId_list = []
jointName_list = []
jointRange_list = []
jointMaxForce_list = []
jointMaxVeloc_list = []
debugId_list = []
temp_debug_value = []
mode = p.POSITION_CONTROL
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(True)
# Constants
g = (0,0,-9.81) 
pi = np.pi

# Setup the environment
print('-'*100)
p.setGravity(*g)
robotId = p.loadURDF(PATH,[0.,0.,initial_height],initial_ori)
planeId = p.loadURDF('plane.urdf')
number_of_joints = p.getNumJoints(robotId)
print(f'Robot id: {robotId}')
print(f'number of robot joints: {number_of_joints}')
for joint_index in range(number_of_joints):
    data = p.getJointInfo(robotId, joint_index)
    jointId_list.append(data[0])                                                                                # Create list to store joint's Id
    jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
    jointRange_list.append((data[8],data[9]))                                                                   # Create list to store joint's Range
    jointMaxForce_list.append(data[10])                                                                         # Create list to store joint's max Force
    jointMaxVeloc_list.append(data[11])                                                                         # Create list to store joint's max Velocity
    p.enableJointForceTorqueSensor(robotId,joint_index,True)
    print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
p.setJointMotorControlArray(robotId,jointId_list,mode)
print(f'Control mode is set to: {"Velocity" if mode==0 else "Position"}')
previous_pos = np.zeros((len(jointId_list)))
print('-'*100)


def leg_traj(t,T,side, mag_thigh = 0.4,mag_bicep=0.4, swing=0.3, scheme = 0):
    noise = np.random.uniform(-T/4,T/4,1)
    print(noise)
    # t += noise
    if scheme == 0:
        if side == 'l':
            return np.hstack([ swing*np.ones_like(t), mag_thigh*np.cos(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])
        else:
            return np.hstack([-swing*np.ones_like(t), mag_thigh*np.cos(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])
    if scheme == 1:
        if side == 'l':
            return np.hstack([ swing*np.ones_like(t), mag_thigh*np.sin(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])
        else:
            return np.hstack([-swing*np.ones_like(t), mag_thigh*np.sin(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])
    if scheme == 2:
        if side == 'l':
            return np.hstack([ swing*np.ones_like(t), -mag_thigh*np.cos(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])
        else:
            return np.hstack([-swing*np.ones_like(t), -mag_thigh*np.cos(2*np.pi*t/T), mag_bicep*np.cos(2*np.pi*t/T)])


def get_run_gait(T,t,scheme):
    t       = np.array(t,np.float32).reshape((-1,1))
    act1    = leg_traj(t,T,'l',scheme=scheme)
    act2    = leg_traj(t+T/2,T,'r',scheme=scheme)
    act3    = leg_traj(t+T/2,T,'l',scheme=scheme)
    act4    = leg_traj(t,T,'r',scheme=scheme)
    action  = np.hstack([act1,act2,act3,act4])
    return action


# Simulation loop
time    = 0
T       = 2*np.pi
num_step= 10 
scheme  = 2
fixed   = False

while True:
    action = get_run_gait(T,time,scheme)
    filtered_action = (previous_pos*.8 + np.array(action)*.2).flatten()
    previous_pos = action
    for _ in range(num_step):
        if fixed :
            p.resetBasePositionAndOrientation(robotId,[0.,0.,1.],initial_ori)
        p.setJointMotorControlArray(robotId,
                                    jointId_list,
                                    mode,
                                    targetPositions = filtered_action,
                                    forces = jointMaxForce_list, 
                                    targetVelocities = jointMaxVeloc_list,
                                    # positionGains = np.ones_like(filtered_action)*.2,
                                    # velocityGains = np.ones_like(temp_debug_value)*0.,        
                                    )
        # p.stepSimulation()
        start   = t.time()
        stop    = start
        while(stop<(start+sleep_time)):
            stop = t.time()
        # t.sleep(sleep_time)
    time += 1