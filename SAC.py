import torch
import torch.nn as nn
import numpy as np
import time as t
import quad_multidirect_env as qe
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
        gamma               = .99,
        zeta                = .5,
        learning_rate       = 1e-4,
        num_robot           = 9,
        epochs              = 500,
        data_size           = 1000,
        batch_size          = 5000,
        reward_index        = np.array([[1, 1, 1, 1, 1, 1]]),
        seed                = 1107,
        device              = None,
        norm                = True,
        terrain_height      = [0, 0.05],
        print_rew           = False,
        real_time           = False,
        train_              = True,
        debug               = False,
        run                 = None,
        temp                = 0.99,
        ):
        
        
        self.PATH               = PATH       
        self.load_model         = load_model
        self.envi               = envi
        self.log_data           = log_data
        self.save_model         = save_model
        self.render_mode        = render_mode
        self.thresh             = thresh
        self.gamma              = gamma 
        self.zeta               = zeta
        self.learning_rate      = learning_rate 
        self.num_robot          = num_robot        
        self.epochs             = epochs
        self.data_size          = data_size
        self.batch_size         = batch_size
        self.reward_index       = reward_index
        self.seed               = seed
        self.device             = device
        self.norm               = norm
        self.terrain_height     = terrain_height
        self.print_rew          = print_rew
        self.real_time          = real_time
        self.train_             = train_
        self.debug              = debug 
        self.run                = run
        self.temp               = temp
        self.static_temp        = 1
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
        self.actor = ACTOR(observation_space,action_space,buffer_length).to(self.device)
        self.critic = CRITIC(observation_space,action_space,buffer_length).to(self.device)
        print('MLP is ready!')
        print('actor params: ',sum(i.numel() for i in self.actor.parameters()) )
        print('critic params: ',sum(i.numel() for i in self.critic.parameters()) )
        
        # Optim setup
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = self.learning_rate)
        if load_model:
            self.actor.load_state_dict(torch.load(self.model_path+"_actor",map_location=self.device))
            self.actor_optimizer.load_state_dict(torch.load(self.optim_path+"_actor",map_location=self.device))
            self.critic.load_state_dict(torch.load(self.model_path+"_critic",map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(self.optim_path+"_critic",map_location=self.device))
        else:
            pass
        self.actor_optimizer.param_groups[0]['lr'] = self.learning_rate
        print('learning rate:',self.actor_optimizer.param_groups[0]['lr'])
        self.critic_optimizer.param_groups[0]['lr'] = self.learning_rate
        print('learning rate:',self.critic_optimizer.param_groups[0]['lr'])
        
        # Tensor board
        if self.log_data:
            self.writer = SummaryWriter(PATH + '//runs//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime()))
    
    
    def get_actor_critic_action_and_quality(self,obs,eval=None,resample=False):
        logits, var = self.actor(obs)
        probs = TanhNormal(loc = np.pi/4*logits, scale=self.zeta*var,max=np.pi/4,min=-np.pi/4)
        if eval is not None:
            action = eval
            quality = self.critic(obs,action)
            return probs.log_prob(action), quality
        elif resample:
            action  = probs.sample()
            quality = self.critic(obs,action)
            return probs.log_prob(action), quality
        else:
            action  = probs.sample()
            return action
    
        
    def get_data_from_env(self,length = None):
        if length:
            pass
        else:
            length=self.data_size
        local_observation   = []
        local_action        = []
        local_reward        = []
        local_info          = []
        observation         = self.env.get_obs()[0]
        for i in range(length) :
            # Act and get observation 
            action                      = self.get_actor_critic_action_and_quality(torch.Tensor(observation).to(self.device))
            action                      = action.cpu()
            local_observation          += [torch.Tensor(observation)]
            local_action               += [torch.Tensor(action)]
            if self.run:
                action = np.array(action) + self.run*self.env.get_run_gait(self.env.time_steps_in_current_episode)
            observation, reward, info   = self.env.sim(np.array(action),real_time=self.real_time,train=self.train_)
            if self.print_rew:
                print(reward)
            reward          = np.sum(reward*self.reward_index,axis=-1)
            # Save var
            local_reward   += [torch.Tensor(reward)]
            local_info     += [torch.Tensor(info)]
        return local_observation, local_action, local_reward, local_info
    
    
    def train(self):
        best_reward = 0
        for epoch in range(self.epochs):
            self.static_temp *= self.temp
            # Sample data from the environment
            with torch.no_grad():
                self.actor = self.actor.eval()
                self.critic= self.critic.eval()
                data = self.get_data_from_env()
            dataset = custom_dataset(data,self.data_size,self.num_robot,self.gamma)
            dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
            
            for iteration, data in enumerate(dataloader):
                self.actor = self.actor.train()
                self.critic= self.critic.train()
                obs, action, reward, next_obs, info = data
                obs, action, reward, next_obs, info = obs.to(self.device), action.to(self.device), reward.to(self.device), next_obs.to(self.device), info.to(self.device)
                logprob, quality                    = self.get_actor_critic_action_and_quality(obs,eval=action)
                with torch.no_grad():
                    self.actor = self.actor.eval()
                    self.critic= self.critic.eval()
                    next_logprob, next_quality      = self.get_actor_critic_action_and_quality(next_obs,resample=True)
                #save model
                if self.save_model:
                    if (quality.mean().item()>best_reward and quality.mean().item() > self.thresh) | ((epoch*(len(dataloader))+iteration) % 250 == 0):
                        best_reward = quality.mean().item()
                        torch.save(self.actor.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+"actor")
                        torch.save(self.actor_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+'_optim_actor')
                        torch.save(self.critic.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+"critic")
                        torch.save(self.critic_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_best_'+str(round(quality.mean().item(),2))+'_optim_critic')
                        print('saved at: '+str(round(quality.mean().item(),2)))
                # Train models
                self.actor = self.actor.train()
                self.critic= self.critic.train()
                # Train critic
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                targetQ     = reward + self.gamma*(1-info)*(next_quality-self.static_temp*logprob)
                TD_residual = targetQ.detach() - quality
                critic_loss = ((TD_residual)**2).mean()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.actor_optimizer.zero_grad()
                new_logprob, new_quality = self.get_actor_critic_action_and_quality(obs,resample=True)
                actor_loss  = (self.static_temp*new_logprob - new_quality).mean()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                # logging info
                if self.log_data:
                    self.writer.add_scalar('Eval/minibatchreward',reward.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Eval/minibatchreturn',quality.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Eval/estreturn',quality.detach().mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/entropy',-logprob.mean().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/criticloss',critic_loss.detach().item(),epoch*(len(dataloader))+iteration)
                    self.writer.add_scalar('Train/actorloss',actor_loss.detach().item(),epoch*(len(dataloader))+iteration)
                print(f'[{epoch}]:[{self.epochs}]|| iter [{epoch*(len(dataloader))+iteration}]: rew: {round(reward.mean().item(),2)} ret: {round(quality.mean().item(),2)} cri: {critic_loss.detach().item()} act: {actor_loss.detach().item()} entr: {-logprob.mean().detach().item()} estqua: {quality.mean().detach().item()}')
        torch.save(self.actor.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+"actor")
        torch.save(self.actor_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_optim_actor')
        torch.save(self.mlp.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+"critic")
        torch.save(self.mlp_optimizer.state_dict(), self.PATH+'models//SAC//'+t.strftime('%Y-%m-%d-%H-%M-%S', t.localtime())+'_optim_critic')
        
class ACTOR(nn.Module):
    def __init__(self,observation_space,action_space,buffer_length):
        super(ACTOR,self).__init__()
        # actor_model
        self.lin_act1= nn.Linear(observation_space*buffer_length,500)
        self.lin_act2= nn.Linear(500,100)
        self.lin_act3= nn.Linear(100,50)
        self.lin_act4= nn.Linear(50,action_space)
        self.lin_act5= nn.Linear(50,action_space)
        nn.init.orthogonal_(self.lin_act1.weight)
        nn.init.orthogonal_(self.lin_act2.weight)
        nn.init.orthogonal_(self.lin_act3.weight)
        nn.init.orthogonal_(self.lin_act4.weight)
        nn.init.orthogonal_(self.lin_act5.weight)
        self.actor_body   = nn.Sequential(
            nn.Flatten(),
            self.lin_act1,
            nn.Tanh(),
            self.lin_act2,
            nn.Tanh(),
            self.lin_act3,
            nn.Tanh(),
        )
    def forward(self,obs):
        latent = self.actor_body(obs)
        mean    = nn.Tanh()(self.lin_act4(latent))
        var     = nn.Sigmoid()(self.lin_act5(latent))
        return mean, var
        
class CRITIC(nn.Module):
    def __init__(self,observation_space,action_space,buffer_length):
        super(CRITIC,self).__init__()
        # critic_model (Q model)
        self.lin_cri1 = nn.Linear(observation_space*buffer_length,500)
        self.lin_cri2 = nn.Linear(action_space,500)   
        self.lin_cri3 = nn.Linear(500,100)
        self.lin_cri4 = nn.Linear(100,50)
        self.lin_cri5 = nn.Linear(50,1)
        nn.init.orthogonal_(self.lin_cri1.weight)
        nn.init.orthogonal_(self.lin_cri2.weight)
        nn.init.orthogonal_(self.lin_cri3.weight)
        nn.init.orthogonal_(self.lin_cri4.weight)
        nn.init.orthogonal_(self.lin_cri5.weight)
        self.critic_body  = nn.Sequential(
            nn.Tanh(),
            self.lin_cri3,
            nn.Tanh(),
            self.lin_cri4,
            nn.Tanh(),
            self.lin_cri5,
        )
    def forward(self,obs,act):
        lat1 = nn.Tanh()(self.lin_cri1(nn.Flatten()(obs)))
        lat2 = nn.Tanh()(self.lin_cri2(act))
        estimate = self.critic_body(lat1+lat2)
        return estimate
        
class custom_dataset(Dataset):
    
    def __init__(self,data,data_size,num_robot,gamma,verbose = False):
        self.data_size  = data_size
        self.num_robot  = num_robot
        self.gamma      = gamma
        self.obs, self.action, self.reward, self.info = data       
        self.local_observation  = torch.hstack(self.obs).reshape((num_robot*data_size,*self.obs[0].shape[1:]))
        self.local_action       = torch.hstack(self.action).reshape((num_robot*data_size,*self.action[0].shape[1:]))
        self.local_reward       = torch.stack(self.reward,dim=1).reshape((num_robot*data_size,*self.reward[0].shape[1:])).view(-1,1)
        self.local_info         = torch.stack(self.info,dim=1).reshape((num_robot*data_size,*self.info[0].shape[1:])).view(-1,1)
        if verbose:
            print(self.local_observation.shape)
            print(self.local_action.shape)
            print(self.local_reward.shape)
            print(self.local_info.shape)
    
    def __len__(self):
        return self.data_size-1
    
    def __getitem__(self, idx):
        return self.local_observation[idx], self.local_action[idx], self.local_reward[idx], self.local_observation[idx+1], self.local_info[idx]

# # # TEST CODE # # #
# trainer = SAC_quad(
#                 num_robot = 9,
#                 learning_rate = 1e-4,
#                 data_size = 20,
#                 batch_size = 10,
#                 epochs=100,
#                 thresh=1,
#                 log_data = False,
#                 save_model = False,
#                 render_mode= None,
#                 run=1,
#                 )
# trainer.train()