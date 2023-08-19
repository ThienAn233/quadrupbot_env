import pybullet as p 
import pybullet_data
import numpy as np
import time as t 

class Quadrup_env():
    def __init__(
        self,
        max_length      = 500,
        num_step        = 10,
        render_mode     = None,
        robot_file      = 'quadrupbot_env\\quadrup.urdf',
        num_robot       = 9,
        terrainHeight   = [0., 0.05],
        seed            = 0,
    ):
        
        # Configurable variables
        self.max_length         = max_length
        self.num_step           = num_step
        if render_mode: 
            self.physicsClient  = p.connect(p.GUI)
        else:
            self.physicsClient  = p.connect(p.DIRECT)
        self.robot_file         = robot_file
        self.num_robot          = num_robot 
        self.target_height      = [0.15, 0.5]
        self.initialVel         = [0, .1]
        self.initialMass        = [0, 1.]
        self.initialPos         = [0, .1]
        self.initialFriction    = [0, .3]
        self.terrainHeight      = terrainHeight
        self.terrainScale       = [.05, .05, 1]
        self.initialHeight      = .2937 + self.terrainHeight[-1]
        self.robotId_list       = []
        self.jointId_list       = []
        self.jointName_list     = []
        self.jointRange_list    = []
        self.jointMaxForce_list = []
        self.jointMaxVeloc_list = []
        
        # Constant DO NOT TOUCH
        self.mode   = p.POSITION_CONTROL
        self.seed   = seed
        np.random.seed(self.seed)
        self.g      = (0,0,-9.81) 
        self.pi     = np.pi
        self.time_steps_in_current_episode = [1 for _ in range(self.num_robot)]
        self.vertical       = np.array([0,0,1])
        self.terrain_shape  = [10, self.num_robot]
        self.feet_list = [2,5,8,11]
        self.base_pos = []
        
        # Setup the environment
        print('-'*100)
        print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId = self.physicsClient)
        p.setGravity(*self.g, physicsClientId = self.physicsClient)
        self.get_init_pos()
        for pos in self.corr_list:
            self.robotId_list      += [p.loadURDF(self.robot_file, physicsClientId = self.physicsClient,basePosition=pos,baseOrientation=[0,0,0,1])]
        self.sample_terrain()
        self.number_of_joints       = p.getNumJoints(self.robotId_list[0], physicsClientId = self.physicsClient)
        for jointIndex in range(0,self.number_of_joints):
            data = p.getJointInfo(self.robotId_list[0], jointIndex, physicsClientId = self.physicsClient)
            self.jointId_list      += [data[0]]                                                                          # Create list to store joint's Id
            self.jointName_list    += [str(data[1])]                                                                     # Create list to store joint's Name
            self.jointRange_list   += [(data[8],data[9])]                                                                # Create list to store joint's Range
            self.jointMaxForce_list+= [data[10]]                                                                         # Create list to store joint's max Force
            self.jointMaxVeloc_list+= [data[11]]                                                                         # Create list to store joint's max Velocity
            print(f'Id: {data[0]}, Name: {str(data[1])}, Range: {(data[8],data[9])}')
        self.robotBaseMassandFriction = [p.getDynamicsInfo(self.robotId_list[0],-1,physicsClientId = self.physicsClient)[0], p.getDynamicsInfo(self.robotId_list[0],self.jointId_list[-1],physicsClientId = self.physicsClient)[1]]
        print(f'Robot mass: {self.robotBaseMassandFriction[0]} and friction on feet: {self.robotBaseMassandFriction[1]}')
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,self.mode, physicsClientId = self.physicsClient)
            for joints in self.jointId_list:
                p.enableJointForceTorqueSensor(robotId,joints,True,physicsClientId = self.physicsClient)
            self.sample_target(robotId)
        print(f'Robot position loaded, force/torque sensors enable')
        self.previous_pos = np.zeros((self.num_robot,len(self.jointId_list)))
        self.reaction_force = np.zeros((self.num_robot,len(self.jointId_list),3))
        print('-'*100) 
        
    
    def get_init_pos(self):
        nrow = int(self.num_robot)
        x = np.linspace(-(nrow-1)/2,(nrow-1)/2,nrow)
        xv,yv = np.meshgrid(0,x)
        xv, yv = np.hstack(xv), np.hstack(yv)
        zv = self.initialHeight*np.ones_like(xv)
        self.corr_list = np.vstack((xv,yv,zv)).transpose()

    
    def sample_target(self,robotId):
        random_Ori  = [0,0,0,1]
        # Sample new position
        pos         = self.corr_list[robotId] + np.array(list(np.random.uniform(*self.initialPos,size=2))+[0])
        p.resetBasePositionAndOrientation(robotId, pos, random_Ori, physicsClientId = self.physicsClient)
        # Sample new velocity
        init_vel    = np.random.normal(loc = self.initialVel[0],scale = self.initialVel[1],size=(3))
        p.resetBaseVelocity(robotId,init_vel,[0,0,0],physicsClientId=self.physicsClient)
        # Sample new base mass
        new_mass    = self.robotBaseMassandFriction[0] + np.random.uniform(*self.initialMass)
        p.changeDynamics(robotId,-1,new_mass)
        # Sample new feet friction
        new_friction= self.robotBaseMassandFriction[1] + np.random.uniform(*self.initialFriction)
        for i in self.feet_list:
            p.changeDynamics(robotId,i,lateralFriction=new_friction)
        for jointId in self.jointId_list:
            p.resetJointState(bodyUniqueId=robotId,jointIndex=jointId,targetValue=0,targetVelocity=0,physicsClientId=self.physicsClient)
                
    
    def sample_terrain(self):
        numHeightfieldRows = int(self.terrain_shape[0]/self.terrainScale[0])
        numHeightfieldColumns = int(self.terrain_shape[1]/self.terrainScale[1])
        heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
        for j in range (int(numHeightfieldColumns/2)):
            for i in range (int(numHeightfieldRows/2) ):
                height = round(np.random.uniform(*self.terrainHeight),2)
                heightfieldData[2*i+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
                heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
                heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
        terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=self.terrainScale, heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, physicsClientId=self.physicsClient)
        self.terrainId = p.createMultiBody(0, terrainShape, physicsClientId=self.physicsClient,useMaximalCoordinates =True)
        p.resetBasePositionAndOrientation(self.terrainId,[+4.5,0,0], [0,0,0,1], physicsClientId=self.physicsClient)
        self.textureId = p.loadTexture("heightmaps/gimp_overlay_out.png")
        p.changeVisualShape(self.terrainId, -1, textureUniqueId = self.textureId)

    
    def get_distance_and_ori_and_velocity_from_target(self,robotId):
        temp_obs_value = []
        # Get cordinate in robot reference 
        base_position, base_orientation =  p.getBasePositionAndOrientation(robotId, physicsClientId = self.physicsClient)[:2]
        self.base_pos = [-base_position[1]+self.corr_list[robotId][1]] + [base_position[-1]]
        temp_obs_value += [ *base_orientation]
        # Get base linear and angular velocity
        linear_velo, angular_velo = p.getBaseVelocity(robotId, physicsClientId = self.physicsClient)
        temp_obs_value += [*linear_velo, *angular_velo]
        return temp_obs_value
    
    
    def get_joints_values(self,robotId):
        temp_obs_value = []
        # Get joints reaction force for reward
        # Get joints position and velocity
        for Id in self.jointId_list:
            temp_obs_value += [*p.getJointState(robotId,Id, physicsClientId = self.physicsClient)[:2]]
            self.reaction_force[robotId,Id,:] = p.getJointState(robotId,Id,physicsClientId = self.physicsClient)[2][:3]
        return temp_obs_value


    def get_contact_values(self,robotId):
        temp_obs_vaule = []
        for link in self.feet_list:
            if p.getContactPoints(robotId,self.terrainId,link):
                temp_obs_vaule += [1.]
            else:
                temp_obs_vaule += [0.]
        return temp_obs_vaule
    

    def get_all_obs(self,robotId):
        temp_obs_value = []
        # Base position state
        base_info = self.get_distance_and_ori_and_velocity_from_target(robotId)
        # Joints state
        joints_info = self.get_joints_values(robotId)
        # Contact state
        contact_info = self.get_contact_values(robotId)
        # Full observation
        temp_obs_value += [
                        *base_info,
                        *joints_info,
                        *contact_info
                        ]
        return temp_obs_value
    
    
    def get_obs(self,train=True):
        
        temp_obs_value = []
        temp_info = []
        temp_reward_value = []

        for robotId in self.robotId_list:
            # GET OBSERVATION
            temp_obs_value += [self.get_all_obs(robotId)]
            # GET INFO
            if train:
                temp_info += [self.auto_reset(robotId)]
            # GET REWARD
            temp_reward_value += [self.get_reward_value(temp_obs_value[-1],robotId)]
        return np.array(temp_obs_value), np.array(temp_reward_value), np.array(temp_info)
    
    
    def truncation_check(self,height,dir,robotId):
        return  (self.time_steps_in_current_episode[robotId]>self.max_length) | (self.target_height[0] > height) | (np.abs(dir)>0.25)
    
    
    def auto_reset(self,robotId):
        height, dir = self.base_pos[1], self.base_pos[0]
        truncation = self.truncation_check(height,dir,robotId)
        if truncation:
            self.sample_target(robotId)
            self.time_steps_in_current_episode[robotId] = 0
            self.previous_pos[robotId] = np.zeros((len(self.jointId_list)))
        return truncation
    
    
    def act(self,action):
        for robotId in self.robotId_list:
            p.setJointMotorControlArray(robotId,self.jointId_list,
                                        self.mode,
                                        targetPositions = action[robotId], 
                                        forces = self.jointMaxForce_list, 
                                        targetVelocities = self.jointMaxVeloc_list, 
                                        positionGains = np.ones_like(self.jointMaxForce_list)*.2,
                                        # velocityGains = np.ones_like(self.jointMaxForce_list)*1,
                                        physicsClientId = self.physicsClient)
    
    
    def sim(self,action,real_time = False,train=True):
        filtered_action = self.previous_pos*.8 + action*.2
        self.previous_pos = action
        self.time_steps_in_current_episode = [self.time_steps_in_current_episode[i]+1 for i in range(self.num_robot)]
        for _ in range(self.num_step):
            self.act(filtered_action)
            p.stepSimulation( physicsClientId = self.physicsClient)
            if real_time:
                t.sleep(self.sleep_time*self.num_step)
        return self.get_obs()
                
                
    def close(self):
        p.disconnect(physicsClientId = self.physicsClient)
        
    
    def get_reward_value(self,obs,robotId):
        # Reward for high speed in x direction
        speed = 25*obs[4]

        # Reward for being in good y direction
        align = -50*self.base_pos[0]**2
        
        # Reward for being high
        high = -100*(-self.base_pos[1]+.28) if self.base_pos[1]<.28 else 0
        
        # Reward for surviving 
        surv = 10
        
        # Reward for minimal force
        force = (-1e-5)*((self.reaction_force[robotId,:]**2).sum())

        return [speed, align, high, surv, force]
    
# # # TEST CODE # # #
# env = Quadrup_env(render_mode='human',num_robot=2,terrainHeight=[0.,0.])             
# for _ in range(1000):
#     action = np.random.uniform(-.1,.1,(env.num_robot,env.number_of_joints))
#     obs, rew, inf = env.sim(action)
#     # t.sleep(.5)
#     print(obs.shape,rew.shape,inf.shape)
#     # print(env.time_steps_in_current_episode)
#     # print(obs[0])
#     # print(rew[0])
#     # print(inf[0])
# env.close()