import torch
import torch.nn as nn
import numpy as np
import time as t
import quad_cnn_env as qe
from torch.utils.data import Dataset, DataLoader
from torchrl.modules import TanhNormal
from torch.utils.tensorboard import SummaryWriter

class SAC_quad():
    def __init__(
        self,
        PATH                = 'quadrupbot_env//',
        load_model          = None,
        envi                = qe,
        log_data            = True,
        save_model          = True,
        render_mode         = False,
        thresh              = 1.5,
        explore             = 1e-4,
        gamma               = .99,
        zeta                = .5,
        learning_rate       = 1e-4,
        num_robot           = 9,
        epochs              = 500,
        data_size           = 1000,
        batch_size          = 5000,
        reward_index        = np.array([[1, 1, 1, 1, 1, 1]]),
        seed                = 1107,
        mlp                 = None,
        device              = None,
        norm                = True,
        terrain_height      = [0, 0.05],
        print_rew           = False,
        real_time           = False,
        train_              = True,
        debug               = False,
        run                 = None,
        ):
        
        
        self.PATH               = PATH       
        self.load_model         = load_model
        self.envi               = envi
        self.log_data           = log_data
        self.save_model         = save_model
        self.render_mode        = render_mode
        self.thresh             = thresh
        self.explore            = explore
        self.gamma              = gamma 
        self.zeta               = zeta
        self.learning_rate      = learning_rate 
        self.num_robot          = num_robot        
        self.epochs             = epochs
        self.data_size          = data_size
        self.batch_size         = batch_size
        self.reward_index       = reward_index
        self.seed               = seed
        self.mlp                = mlp
        self.device             = device
        self.norm               = norm
        self.terrain_height     = terrain_height
        self.print_rew          = print_rew
        self.real_time          = real_time
        self.train_             = train_
        self.debug              = debug 
        self.run                = run
        
        # Setup random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        print(f'Using seed: {self.seed}')
        
        # Load model path and device
        if load_model:
            self.model_path = PATH + '//models//SAC//' + load_model
            self.optim_path = PATH + '//models//SAC//' + load_model + '_optim'
        if self.device:
            pass
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('Using device: ', self.device)
        
        # Setup env
        self.env = envi.Quadrup_env(num_robot=self.num_robot,render_mode=self.render_mode,terrainHeight=self.terrain_height,debug=self.debug)
        print('env is ready!')
        action_space            = self.env.action_space     
        observation_space       = self.env.observation_space
        buffer_length           = self.env.buffer_length
        self.action_space       = action_space     
        self.observation_space  = observation_space
        print(f'action space of         {num_robot} robot is: {self.action_space}')
        print(f'observation sapce of    {num_robot} robot is: {self.observation_space}')
        
        # Setup MLP
        if self.mlp:
            self.mlp.to(self.device)
            pass
        else:
            class MLP(nn.Module):
                def __init__(self):
                    super(MLP,self).__init__()
                    siz = (observation_space//2)*(buffer_length-1-2-4-8-16)
                    # nn setup
                    # policy network
                    policy_conv1= nn.Conv1d(observation_space,observation_space*8,2,dilation=1)
                    policy_conv2= nn.Conv1d(observation_space*8,observation_space*4,2,dilation=2)
                    policy_conv3= nn.Conv1d(observation_space*4,observation_space*2,2,dilation=4)
                    policy_conv4= nn.Conv1d(observation_space*2,observation_space,2,dilation=8)
                    policy_conv5= nn.Conv1d(observation_space,observation_space//2,2,dilation=16)
                    self.policy_pre_pros = nn.Sequential(
                        policy_conv1,
                        nn.LeakyReLU(.2),
                        policy_conv2,
                        nn.LeakyReLU(.2),
                        policy_conv3,
                        nn.LeakyReLU(.2),
                        policy_conv4,
                        nn.LeakyReLU(.2),
                        policy_conv5,
                        nn.LeakyReLU(.2),
                        nn.Flatten()
                    )
                    policy_lin1 = nn.Linear(siz,500)
                    torch.nn.init.xavier_normal_(policy_lin1.weight,gain=1)
                    policy_lin2 = nn.Linear(500,100)
                    torch.nn.init.xavier_normal_(policy_lin2.weight,gain=1)
                    # quality network
                    quality_conv1= nn.Conv1d(observation_space,observation_space*8,2,dilation=1)
                    quality_conv2= nn.Conv1d(observation_space*8,observation_space*4,2,dilation=2)
                    quality_conv3= nn.Conv1d(observation_space*4,observation_space*2,2,dilation=4)
                    quality_conv4= nn.Conv1d(observation_space*2,observation_space,2,dilation=8)
                    quality_conv5= nn.Conv1d(observation_space,observation_space//2,2,dilation=16)
                    self.quality_pre_pros = nn.Sequential(
                        quality_conv1,
                        nn.LeakyReLU(.2),
                        quality_conv2,
                        nn.LeakyReLU(.2),
                        quality_conv3,
                        nn.LeakyReLU(.2),
                        quality_conv4,
                        nn.LeakyReLU(.2),
                        quality_conv5,
                        nn.LeakyReLU(.2),
                        nn.Flatten()
                    )
                    
                    quality_lin1 = nn.Linear(siz,500)
                    torch.nn.init.xavier_normal_(quality_lin1.weight,gain=1)
                    quality_lin2 = nn.Linear(500,100)
                    torch.nn.init.xavier_normal_(quality_lin2.weight,gain=1)
                    lin3 = nn.Linear(100,action_space)
                    torch.nn.init.xavier_normal_(lin3.weight,gain=0.2)
                    lin4 = nn.Linear(100,action_space)
                    torch.nn.init.constant_(lin4.weight,1.)
                    lin5 = nn.Linear(100,1)
                    torch.nn.init.xavier_normal_(lin5.weight,gain=1)
                    
                    self.mean = nn.Sequential(
                        self.policy_pre_pros,
                        policy_lin1,
                        nn.LeakyReLU(.2),
                        policy_lin2,
                        nn.LeakyReLU(.2),
                        lin3,
                    )
                    self.var = nn.Sequential(
                        self.policy_pre_pros,
                        policy_lin1,
                        nn.LeakyReLU(.2),
                        policy_lin2,
                        nn.LeakyReLU(.2),
                        lin4,
                    )
                    self.critic = nn.Sequential(
                        self.quality_pre_pros,
                        quality_lin1,
                        nn.LeakyReLU(.2),
                        quality_lin2,
                        nn.LeakyReLU(.2),
                        lin5,
                    )
                def forward(self,input):
                    return self.mean(input),self.var(input),self.critic(input)
            self.mlp = MLP().to(self.device)
        print('MLP is ready!')
        print('params: ',sum(i.numel() for i in self.mlp.parameters()) )
        

            
        # Optim setup
        self.mlp_optimizer = torch.optim.Adam(self.mlp.parameters(),lr = self.learning_rate)
        if load_model:
            self.mlp.load_state_dict(torch.load(self.model_path,map_location=self.device))
            self.mlp_optimizer.load_state_dict(torch.load(self.optim_path,map_location=self.device))
        else:
            pass
        self.mlp_optimizer.param_groups[0]['lr'] = self.learning_rate
        print('learning rate:',self.mlp_optimizer.param_groups[0]['lr'])
        
        # Tensor board
        if self.log_data:
            self.writer = SummaryWriter(PATH + '//runs//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
    
    
    def get_actor_critic_action_and_quality(self,obs,eval=True):
        logits, var, quality = self.mlp(obs)
        old_shape = logits.shape
        logits, var = logits.view(*logits.shape,1), var.view(*var.shape,1)
        probs = TanhNormal(loc = logits, scale=self.zeta*nn.Sigmoid()(var),max=np.pi/4,min=-np.pi/4)
        if eval is True:
            action = probs.sample()
            return action.view(old_shape), probs.log_prob(action.view(logits.shape)).prod(dim=-1), quality
        else:
            action = eval
            return action.view(old_shape), probs.log_prob(action.view(logits.shape)).prod(dim=-1), quality
    
        
    def get_data_from_env(self,length = None):
        if length:
            pass
        else:
            length=self.data_size
        local_observation   = []
        local_action        = []
        local_reward        = []
        local_timestep      = []
        observation         = self.env.get_obs()[0]
        local_timestep      = []
        for i in range(length) :
            
            # Act and get observation 
            timestep                    = np.array(self.env.time_steps_in_current_episode)
            local_timestep             +=[torch.Tensor(timestep.copy())]
            action, logprob, _          = self.get_actor_critic_action_and_quality(torch.Tensor(observation).to(self.device))
            action, logprob             = action.cpu(), logprob.cpu()
            local_observation          += [torch.Tensor(observation)]
            local_action               += [torch.Tensor(action)]
            if self.run:
                action = np.array(action) + self.run*self.env.get_run_gait(self.env.time_steps_in_current_episode)
            observation, reward, info   = self.env.sim(np.array(action),real_time=self.real_time,train=self.train_)
            if self.print_rew:
                print(reward)
            reward          = np.sum(reward*self.reward_index,axis=-1) - logprob.numpy()
            
            # Save var
            local_reward   += [torch.Tensor(reward)]
        return local_observation, local_action, local_reward, local_timestep
    
    
    def train(self):
        best_reward = 0
        for epoch in range(self.epochs):
            mlp = self.mlp.eval()
            
            # Sample data from the environment
            with torch.no_grad():
                data = self.get_data_from_env()
            dataset = custom_dataset(data,self.data_size,self.num_robot,self.gamma,verbose=True)
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            
            for iteration, data in enumerate(dataloader):
                mlp = mlp.train()
                obs, action, reward, next_obs           = data
                obs, action, reward, next_obs           = obs.to(self.device), action.to(self.device), reward.to(self.device), next_obs.to(self.device)
                _, logprob, quality                     = self.get_actor_critic_action_and_quality(obs,eval=action)
                with torch.no_grad():
                    next_action, next_logprob, next_quality = self.get_actor_critic_action_and_quality(next_obs)
                # Train models
                self.mlp_optimizer.zero_grad()
                TD_residual = reward + self.gamma*next_quality - quality 
                critic_loss = .5*((TD_residual)**2).mean()
                if epoch % 4 == 3:
                    actor_loss  = nn.KLDivLoss(reduction="batchmean",log_target=True)(next_logprob,nn.functional.log_softmax(quality,dim=1).detach())
                else:
                    actor_loss  = torch.tensor(0)
                loss        = critic_loss + actor_loss
                loss.backward()
                self.mlp_optimizer.step()
                
                #save model
                if self.save_model:
                    if (quality.mean().item()>best_reward and quality.mean().item() > self.thresh) | ((epoch*(len(dataloader))+iteration) % 250 == 0):
                        best_reward = quality.mean().item()
                        torch.save(mlp.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2)))
                        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+'_optim')
                        print('saved at: '+str(round(quality.mean().item(),2)))
                
                # logging info
                if self.log_data:
                    self.writer.add_scalar('Eval/minibatchreward',reward.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Eval/minibatchreturn',quality.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/criticloss',critic_loss.detach().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/actorloss',actor_loss.detach().item(),epoch*(len(dataloader))+iteration)
                print(f'[{epoch}]:[{self.epochs}]|| iter [{epoch*(len(dataloader))+iteration}]: rew: {round(reward.mean().item(),2)} ret: {round(quality.mean().item(),2)} cri: {critic_loss.detach().item()} act: {actor_loss.detach().item()}')
        torch.save(mlp.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_optim')
        

class custom_dataset(Dataset):
    
    def __init__(self,data,data_size,num_robot,gamma,verbose = False):
        self.data_size  = data_size
        self.num_robot  = num_robot
        self.gamma      = gamma
        self.obs, self.action, self.reward, self.timestep = data       
        self.local_return = [0 for i in range(data_size)]
        self.calculate_return()
        self.local_return = torch.hstack(self.local_return).reshape((num_robot*data_size,*self.local_return[0].shape[1:])).view(-1,1)
        self.local_observation = torch.hstack(self.obs).reshape((num_robot*data_size,*self.obs[0].shape[1:]))
        self.local_action = torch.hstack(self.action).reshape((num_robot*data_size,*self.action[0].shape[1:]))
        self.local_reward = torch.stack(self.reward,dim=1).reshape((num_robot*data_size,*self.reward[0].shape[1:])).view(-1,1)
        self.local_timestep = torch.stack(self.timestep,dim=1).reshape((num_robot*data_size,*self.timestep[0].shape[1:])).view(-1,1)
        self.check_time()
        if verbose:
            print(self.local_return.shape)
            print(self.local_observation.shape)
            print(self.local_action.shape)
            print(self.local_reward.shape)
            print(self.local_timestep.shape)

    def isnt_end(self, i):
        ### THE FIRST EPS WILL BE TIMESTEP 1, THE FINAL EP WILL BE TIMESTEP 0
        return self.timestep[i] != 0
    
    def calculate_return(self):
        for i in range(self.data_size-1,-1,-1):
            if i == self.data_size-1 :
                self.local_return[i] = self.reward[i]
            else:
                self.local_return[i] = self.reward[i] + self.isnt_end(i)*self.gamma*self.local_return[i+1]
        return self.local_return
    
    def check_time(self):
        self.index = []
        for index in range(len(self.local_timestep)):
            if index == len(self.local_timestep)-1:
                continue
            elif self.local_timestep[index][0] < self.local_timestep[index+1][0] :
                self.index += [index]
            else:
                continue
        return self.index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        idx = self.index[index]
        return self.local_observation[idx], self.local_action[idx], self.local_reward[idx], self.local_observation[idx+1]

# # # TEST CODE # # #
trainer = SAC_quad(
                num_robot = 9,
                learning_rate = 1e-4,
                data_size = 20,
                batch_size = 10,
                epochs=100,
                thresh=1,
                explore = 1e-2,
                log_data = False,
                save_model = False,
                render_mode= True,
                run=1,
                )
trainer.train()