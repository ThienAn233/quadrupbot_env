import pybullet as p 
import pybullet_data
import numpy as np
import time as t 
import utils

class Quadrup_env():
    def __init__(
        self,
        max_length      = 500,
        num_step        = 10,
        render_mode     = None,
        debug           = False,
        robot_file      = 'quadrupbot_env//quadrup.urdf',
        num_robot       = 9,
        terrainHeight   = [0., 0.05],
        seed            = 0,
        buffer_length   = 60,
    ):
        # Configurable variables
        self.clientId           = []
        for i in range(num_robot):
            if isinstance(render_mode, str) and i == 0: 
                self.clientId  += [p.connect(p.GUI)]
            elif i == render_mode:
                self.clientId  += [p.connect(p.GUI)]
            else:
                self.clientId  += [p.connect(p.DIRECT)]
        self.max_length         = max_length
        self.num_step           = num_step
        self.debug              = debug
        self.robot_file         = robot_file
        self.num_robot          = num_robot 
        self.initialVel         = [0, .2]
        self.initialMass        = [0, 1.]
        self.initialPos         = [0, .3]
        self.initialFriction    = [0.8, 1.5]
        self.terrainHeight      = terrainHeight
        self.buffer_length      = buffer_length
        self.terrainScale       = [.05, .05, 1]
        self.initialHeight      = .2937
        self.jointId_list       = []
        self.jointName_list     = []
        self.jointRange_list    = []
        self.jointMaxForce_list = []
        self.jointMaxVeloc_list = []
        
        # Constant DO NOT TOUCH
        self.robotId= 0
        self.mode   = p.POSITION_CONTROL
        self.seed   = seed
        self.sleep_time = 1./240.
        np.random.seed(self.seed)
        self.g      = (0,0,-9.81) 
        self.pi     = np.pi
        self.T      = self.pi
        self.time_steps_in_current_episode = [1 for _ in range(self.num_robot)]
        self.vertical       = np.array([0,0,1])
        self.terrain_shape  = [20, 20]
        self.feet_list = [2,5,8,11]
        self.control_body       = np.zeros((self.num_robot,3))
        self.control_face       = np.zeros((self.num_robot,3))
        self.base_pos           = np.zeros((self.num_robot,3))
        self.base_ori           = np.zeros((self.num_robot,3))
        self.base_lin_vel       = np.zeros((self.num_robot,3))
        self.base_ang_vel       = np.zeros((self.num_robot,3))
        self.target_dir_world   = np.zeros((self.num_robot,3))
        self.target_vel_world   = np.zeros((self.num_robot,3))
        self.target_dir_robot   = np.zeros((self.num_robot,3))
        self.target_vel_robot   = np.zeros((self.num_robot,3))
        
        
        # Setup the environment
        print('-'*100)
        print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        for client in self.clientId:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
            p.setGravity(*self.g, physicsClientId=client)
            p.loadURDF(self.robot_file, basePosition=[0,0,self.initialHeight],baseOrientation=[0,0,0,1], physicsClientId=client)
        self.sample_terrain()
        self.number_of_joints = p.getNumJoints(self.robotId, physicsClientId = self.clientId[0])
        for jointIndex in range(0,self.number_of_joints):
            data = p.getJointInfo(self.robotId, jointIndex, physicsClientId = self.clientId[0])
            self.jointId_list      += [data[0]]                                                                          # Create list to store joint's Id
            self.jointName_list    += [str(data[1])]                                                                     # Create list to store joint's Name
            self.jointRange_list   += [(data[8],data[9])]                                                                # Create list to store joint's Range
            self.jointMaxForce_list+= [data[10]]                                                                         # Create list to store joint's max Force
            self.jointMaxVeloc_list+= [data[11]]                                                                         # Create list to store joint's max Velocity
            print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
        self.robotBaseMassandFriction = [p.getDynamicsInfo(self.robotId,-1,physicsClientId = self.clientId[0])[0], p.getDynamicsInfo(self.robotId,self.jointId_list[-1],physicsClientId = self.clientId[0])[1]]
        print(f'Robot mass: {self.robotBaseMassandFriction[0]} and friction on feet: {self.robotBaseMassandFriction[1]}')
        for client in self.clientId:
            p.setJointMotorControlArray(self.robotId,self.jointId_list,self.mode, physicsClientId = client)
            for joints in self.jointId_list:
                p.enableJointForceTorqueSensor(self.robotId,joints,True,physicsClientId = client)
            self.sample_target(client)
            print(f'Sampled for client {client}')
        print(f'Robot position loaded, force/torque sensors enable')
        self.previous_pos       = np.zeros((self.num_robot,len(self.jointId_list)))
        self.reaction_force     = np.zeros((self.num_robot,len(self.jointId_list),3))
        self.contact_force      = np.zeros((self.num_robot,len(self.feet_list))) 
        self.action_space       = len(self.jointId_list)
        self.observation_space  = len(self.get_all_obs(0))
        self.obs_buffer         = np.zeros((self.num_robot,self.observation_space,self.buffer_length))
        print(f'Observation space:  {self.observation_space}')
        print(f'Action space:       {self.action_space}')
        for client in self.clientId:
            self.reset_buffer(client)
        return
    
    
    def reset_buffer(self,client):
        self.obs_buffer[client] = np.array([self.get_all_obs(client)]).T*np.ones((self.observation_space,self.buffer_length))
    
    
    def update_buffer(self,client):
        new_obs = np.array([self.get_all_obs(client)]).T
        self.obs_buffer[client] = np.concatenate((new_obs,self.obs_buffer[client,:,0:-1]),axis=-1)
    
    
    def sample_target(self,client):
        random_Ori  = [0,0,0,1]
        # Sample new position
        pos         = np.array(list(np.random.uniform(*self.initialPos,size=2))+[self.initialHeight+self.terrainHeight[-1]])
        p.resetBasePositionAndOrientation(self.robotId, pos, random_Ori, physicsClientId = client)
        # Sample new velocity
        init_vel    = np.random.normal(loc = self.initialVel[0],scale = self.initialVel[1],size=(3))
        p.resetBaseVelocity(self.robotId,init_vel,[0,0,0],physicsClientId=client)
        # Sample new base mass
        new_mass    = self.robotBaseMassandFriction[0] + np.random.uniform(*self.initialMass)
        p.changeDynamics(self.robotId,-1,new_mass,physicsClientId=client)
        # Sample new feet friction
        new_friction= np.random.uniform(*self.initialFriction)
        for i in self.feet_list:
            p.changeDynamics(self.robotId,i,lateralFriction=new_friction,physicsClientId=client)
        for jointId in self.jointId_list:
            p.resetJointState(bodyUniqueId=self.robotId,jointIndex=jointId,targetValue=0,targetVelocity=0,physicsClientId=client)
        # Sample target direction and velocity
        new_direction   = np.random.normal(0,1,3)
        new_velodir     = np.random.normal(0,1,3)
        new_direction, new_velodir = new_direction/np.linalg.norm(new_direction), new_velodir/np.linalg.norm(new_velodir)
        self.target_dir_world[client] = new_direction
        self.target_vel_world[client] = new_velodir
        return
    
    
    def sample_terrain(self):
        numHeightfieldRows = int(self.terrain_shape[0]/(self.terrainScale[0]))
        numHeightfieldColumns = int(self.terrain_shape[1]/(self.terrainScale[1]))
        for i in range(len(self.clientId)):
            x = np.linspace(-np.pi/2,self.terrain_shape[0],numHeightfieldRows)
            y = np.linspace(-np.pi/2,self.terrain_shape[1],numHeightfieldColumns)
            xx, yy = np.meshgrid(x,y)
            zz = np.random.uniform(*self.terrainHeight,(numHeightfieldColumns,numHeightfieldRows))
            heightfieldData = zz.flatten()
            terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=self.terrainScale, heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.clientId[i])
            self.terrainId = p.createMultiBody(0, terrainShape, useMaximalCoordinates=True, physicsClientId=self.clientId[i])
            p.resetBasePositionAndOrientation(self.terrainId,[0,0,0], [0,0,0,1], physicsClientId=self.clientId[i])
            self.textureId = p.loadTexture('quadrupbot_env//color_map.png',physicsClientId=self.clientId[i])
            p.changeVisualShape(self.terrainId, -1, textureUniqueId = self.textureId,physicsClientId=self.clientId[i])
            p.changeVisualShape(self.terrainId, -1, rgbaColor=[1,1,1,1],physicsClientId=self.clientId[i])

    
    def get_distance_and_ori_and_velocity_from_target(self,client):
        temp_obs_value = []
        # Get cordinate in robot reference 
        vec = np.array([0,0,1,0])
        base_position, base_orientation =  p.getBasePositionAndOrientation(self.robotId,physicsClientId=client)[:2]
        local_orientation = utils.passive_rotation(np.array(base_orientation),vec)[:3]
        self.base_pos[client,:] = np.array(base_position)
        self.base_ori[client,:] = np.array(local_orientation)
        temp_obs_value += [ *local_orientation]
        # Get base linear and angular velocity
        linear_velo, angular_velo = p.getBaseVelocity(self.robotId,physicsClientId=client)
        linear_velo, angular_velo = np.array(list(linear_velo)+[1]), np.array(list(angular_velo)+[1]) 
        linear_velo, angular_velo = utils.active_rotation(np.array(base_orientation),linear_velo)[:3], utils.active_rotation(np.array(base_orientation),angular_velo)[:3]
        temp_obs_value += [*linear_velo, *angular_velo]
        self.base_lin_vel[client,:] = linear_velo
        self.base_ang_vel[client,:] = angular_velo
        return temp_obs_value
    
           
    def get_joints_values(self,client):
        temp_obs_value = []
        # Get joints reaction force for reward
        # Get joints position and velocity
        for Id in self.jointId_list:
            temp_obs_value += [*p.getJointState(self.robotId,Id, physicsClientId=client)[:2]]
            self.reaction_force[client,Id,:] = p.getJointState(self.robotId,Id,physicsClientId =client)[2][:3]
        return temp_obs_value     
    
    
    def get_contact_values(self,client):
        temp_obs_vaule = []
        for i, link in enumerate(self.feet_list):
            if p.getContactPoints(self.robotId,self.terrainId,link,physicsClientId =client):
                temp_obs_vaule += [1.]
                self.contact_force[client,i] = p.getContactPoints(self.robotId,self.terrainId,link,physicsClientId =client)[0][9]
            else:
                temp_obs_vaule += [0.]
                self.contact_force[client,i] = 0.
        return temp_obs_vaule
    
    
    def calculate_target(self,client):
        temp_obs_vaule = []
        base_pos, base_orientation =  p.getBasePositionAndOrientation(self.robotId,physicsClientId =client)
        # Calculate target direction
        target_dir = self.target_dir_world[client] + base_pos
        target_dir = np.array(list(target_dir)+[0])
        target_dir = target_dir/np.linalg.norm(target_dir)
        target_dir = utils.active_rotation(np.array(base_orientation),target_dir)[:3]
        target_dir = np.array([target_dir[0],target_dir[1],0])
        target_dir = target_dir/np.linalg.norm(target_dir)
        self.target_dir_robot[client] = target_dir
        temp_obs_vaule += [*target_dir]
        # Calculate target velodirect
        target_vel = self.target_vel_world[client] + base_pos
        target_vel = np.array(list(target_vel)+[0])
        target_vel = target_vel/np.linalg.norm(target_vel)
        target_vel = utils.active_rotation(np.array(base_orientation),target_vel)[:3]
        target_vel = np.array([target_vel[0],target_vel[1],0])
        target_vel = target_vel/np.linalg.norm(target_vel)
        self.target_vel_robot[client] = target_vel
        temp_obs_vaule += [*target_vel]
        return temp_obs_vaule
    
    
    def get_previous_action(self,client):
        temp_obs_value = []
        temp_obs_value += [*self.previous_pos[client]]
        return temp_obs_value
    
    
    def get_all_obs(self,client):
        temp_obs_value = []
        # Base position state
        base_info       = self.get_distance_and_ori_and_velocity_from_target(client)
        # Joints state
        joints_info     = self.get_joints_values(client)
        # Contact state
        contact_info    = self.get_contact_values(client)
        # Previous action
        previous_action = self.get_previous_action(client)
        # Target state
        target_info     = self.calculate_target(client)
        # Full observation
        temp_obs_value += [
                        *base_info,
                        *joints_info,
                        # *contact_info,
                        *previous_action,
                        *target_info,
                        ]
        return temp_obs_value
    
    
    def truncation_check(self,ori,client):
        return  (self.time_steps_in_current_episode[client]>self.max_length) | (ori<.5)
    
    
    def auto_reset(self,client):
        ori = np.sum(self.base_ori[client][-1])/np.linalg.norm(self.base_ori[client])
        truncation = self.truncation_check(ori,client)
        if truncation:
            self.sample_target(client)
            self.reset_buffer(client)
            self.time_steps_in_current_episode[client] = 0
            self.previous_pos[client] = np.zeros((len(self.jointId_list)))
        return truncation
    
    
    def get_obs(self,train=True):
        
        temp_info = []
        temp_reward_value = []

        for client in self.clientId:
            # GET OBSERVATION
            self.update_buffer(client)
            # GET REWARD
            temp_reward_value += [self.get_reward_value(client)]
            # GET INFO
            if train:
                temp_info += [self.auto_reset(client)]
        return self.obs_buffer, np.array(temp_reward_value), np.array(temp_info)
    
    
    def act(self,action):
        for client in self.clientId:
            p.setJointMotorControlArray(self.robotId,
                                        self.jointId_list,
                                        self.mode,
                                        targetPositions = action[client], 
                                        forces = self.jointMaxForce_list, 
                                        targetVelocities = self.jointMaxVeloc_list, 
                                        # positionGains = np.ones_like(self.jointMaxForce_list)*.2,
                                        # velocityGains = np.ones_like(self.jointMaxForce_list)*1,
                                        physicsClientId = client)
    
    
    def sim(self,action,real_time = False,train=True):
        filtered_action = self.previous_pos*.8 + action*.2
        self.previous_pos = action
        self.time_steps_in_current_episode = [self.time_steps_in_current_episode[i]+1 for i in range(self.num_robot)]
        for _ in range(self.num_step):
            self.act(filtered_action)
            for client in self.clientId:
                p.stepSimulation( physicsClientId=client)
            if real_time:
                t.sleep(self.sleep_time)
        if self.debug:
            self.viz()
        return self.get_obs(train)
    
    
    def viz(self):
        for client in self.clientId:
            self.viz_ori(client)
            self.viz_target(client)
            
            
    def viz_ori(self,client):
        base_pos = self.base_pos[client]
        base_ori = self.base_ori[client]
        p.addUserDebugLine(base_pos,base_pos+base_ori/5,lineWidth = 2, lifeTime =.5, lineColorRGB = [0,0,0],physicsClientId = client)
        return
    
    
    def viz_target(self,client):
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+np.array([0,0,0.2]),lineWidth = 2, lifeTime =.5, lineColorRGB = [0,0,1],physicsClientId = client)
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+np.array([0,0.2,0]),lineWidth = 2, lifeTime =.5, lineColorRGB = [0,1,0],physicsClientId = client)
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+np.array([0.2,0,0]),lineWidth = 2, lifeTime =.5, lineColorRGB = [1,0,0],physicsClientId = client)
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+self.base_lin_vel[client],lineWidth = 2, lifeTime =.5, lineColorRGB = [0,0,0],physicsClientId = client)
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+self.target_vel_robot[client],lineWidth = 2, lifeTime =.5, lineColorRGB = [1,1,1],physicsClientId = client)
        p.addUserDebugLine([0,0,1],np.array([0,0,1])+self.target_dir_robot[client],lineWidth = 2, lifeTime =.5, lineColorRGB = [.5,.5,.5],physicsClientId = client)
        return
    
    
    def leg_traj(self,t,mag_thigh = 0.3,mag_bicep=0.3):
        return np.hstack([np.zeros_like(t), mag_thigh*np.cos(2*np.pi*t/self.T), mag_bicep*np.cos(2*np.pi*t/self.T)])

    
    def get_run_gait(self,t):
        t       = np.array(t).reshape((-1,1))
        act1    = self.leg_traj(t)
        act2    = self.leg_traj(t+self.T/2)
        action  = np.hstack([act1,act2,act2,act1])
        return action
    
    
    def get_reward_value(self,client):
        # Reward for high speed in target velocity direction
        velo_vec = np.sum(self.base_lin_vel[client]*self.target_vel_robot[client])/np.linalg.norm(self.target_dir_robot[client])
        speed = 5*velo_vec

        # Reward for being in good target direction
        align_vec = np.sum(self.target_dir_robot[client][0])/np.linalg.norm(self.target_dir_robot[client])
        align = align_vec
        
        # Reward for termination
        ori = np.sum(self.base_ori[client][-1])/np.linalg.norm(self.base_ori[client])
        high = -20 if ori < .5 else 0
        
        # Reward for surviving 
        surv = 0.1
        
        # Reward for minimal force
        force = (-1e-7)*((self.reaction_force[client,:]**2).sum())

        # Reward for minimal contact force
        contact =(-1e-6)*((self.contact_force[client,:]**2).sum())
        
        return [speed, align, high, surv, force,  contact]
    
# # # TEST CODE # # #
# env = Quadrup_env(  render_mode     = 'human',
#                     num_robot       = 2,
#                     debug           = True,
#                     terrainHeight   = [0. ,0.05],
#                   )
# for time in range(1000):
#     # print(env.time_steps_in_current_episode)
#     action = env.get_run_gait(env.time_steps_in_current_episode)
#     # action = np.random.uniform(-.1,.1,(env.num_robot,env.number_of_joints))
#     obs, rew, inf = env.sim(action,real_time=True)
#     # t.sleep(1./240.)
#     # t.sleep(.5)
#     # print(obs.shape,rew.shape,inf.shape)
#     # print(env.time_steps_in_current_episode)
#     # print(time,obs[0])
#     # print(rew[0])
#     # print(inf[0])
#     # print('-'*100)