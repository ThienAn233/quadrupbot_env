# from quad_multi_direct_v3_gym import Quadrup_env
from quad_multi_real_bot_gym import Quadrup_env
# from quad_multi_real_bot_dummy import Quadrup_env
import pybullet as p
import numpy as np
from stable_baselines3 import SAC
import time as t
from vpython import *
buffer_len = 300
interval = 1./30.
R = 10
phi = 0
n_change = 0.04
v_change = 0.1
plot = False

### reward graph
if plot:
    g1 = graph(title='Reward Graph', xtitle='Reward value', ytitle='Time step')
    r_name = ['align', 'speed', 'high', 'surv', 'force',  'contact']
    r_show = [[(i*interval,0.) for i in range(buffer_len)] for i in range(len(r_name)+1)]
    curv_list = [gcurve(color = .5*(vec.random()+vec(1,1,1)),label=name,legend=True,graph=g1) for name in r_name]
    curv_list.append(gcurve(color = .5*(vec.random()+vec(1,1,1)),label='sum',legend=True,graph=g1))


    ### action graph
    # g2 = graph(title='Ref gait Graph', xtitle='Signal', ytitle='Time step',width=1280)
    # g3 = graph(title='Control Graph', xtitle='Signal', ytitle='Time step',width=1280)
    # a_show = [[(i*interval,0.)for i in range(buffer_len)] for i in range(12)]
    # ref_show = [[(i*interval,0.)for i in range(buffer_len)] for i in range(12)]
    # cmap = [.5*(vec.random()+vec(1,1,1)) for i in range(12)]
    # curv_list2 = [gcurve(color = cmap[i],label='joint '+str(i),legend=True,graph=g2) for i in range(12)]
    # curv_list3 = [gcurve(color = cmap[i],label='joint '+str(i),legend=True,graph=g3) for i in range(12)]


env = Quadrup_env(render_mode = 'human',buffer_length=5,ray_test=False,noise =0,terrain_type=3,terrainHeight=[0,0.025],seed=1,max_length=2000,)
# model = SAC.load('SAC_gym_2024-02-27-11-23-48',device='cpu',print_system_info=True)
# model = SAC.load('SAC_real_gym_2024-05-03-15-57-10',device='cpu',print_system_info=True)
# model = SAC.load('SAC_real_gym_2024-05-12-06-39-22',device='cpu',print_system_info=True) # Trot gait
model = SAC.load('SAC_real_gym_2024-05-27-05-14-04',device='cpu',print_system_info=True) # Bound gait
obs, info = env.reset()
env.target_dir_world[0] = np.array([R*np.cos(phi),R*sin(phi),0.2937])
time = buffer_len*interval+interval
now = t.time()


id1 = p.addUserDebugParameter('_dis', rangeMin = -2, rangeMax = 2 )
id2 = p.addUserDebugParameter('_yaw', rangeMin = -180, rangeMax = 180 )
id3 = p.addUserDebugParameter('_pit', rangeMin = -60, rangeMax = 30 )


while True:
    rate(30)
    disdeb = p.readUserDebugParameter(id1)
    yawdeb = p.readUserDebugParameter(id2)
    pitdeb = p.readUserDebugParameter(id3)
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action,real_time=False)
    ref = env.get_run_gait(env.time_steps_in_current_episode[0])
    pos = env.base_pos[0]
    dir = env.target_dir_world[0]-env.base_pos[0]
    yaw = np.arctan2(dir[1],dir[0])
    p.resetDebugVisualizerCamera(2+disdeb,180*yaw/np.pi-90+yawdeb,-30+pitdeb,pos)
    
    
    keys = p.getKeyboardEvents()
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW]&p.KEY_IS_DOWN:
        phi +=n_change
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW]&p.KEY_IS_DOWN:
        phi -=n_change
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW]&p.KEY_IS_DOWN:
        R +=v_change
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW]&p.KEY_IS_DOWN:
        R -=v_change
    x_corr, y_corr = R*np.cos(phi)+env.base_pos[0][0], R*np.sin(phi)+env.base_pos[0][1]
    env.target_dir_world[0] = np.array([x_corr,y_corr,0.2937])
    
    
    # if terminated or truncated:
    if p.B3G_PAGE_UP in keys and keys[p.B3G_PAGE_UP]&p.KEY_IS_DOWN:
        obs, info = env.reset()
    if (t.time()-now)>interval:
        
        
        # plot reward
        if plot:
            for i,name in enumerate(r_name):
                r_show[i].append((time,info[name]))
                r_show[i].pop(0)
                curv_list[i].data = r_show[i]
            r_show[-1].append((time,reward))
            r_show[-1].pop(0)
            curv_list[-1].data = r_show[-1]
        
        
            # plot action
            # for i in range(12):
            #     a_show[i].append((time,action[i]))
            #     a_show[i].pop(0)
            #     curv_list2[i].data = a_show[i]

            #     ref_show[i].append((time,ref.flatten()[i]))
            #     ref_show[i].pop(0)
            #     curv_list3[i].data = ref_show[i]


        time += interval
        now = t.time()