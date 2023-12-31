import PPO
import quad_environment as qa

eval = PPO.PPO_quad(
    load_model      = '2023-09-08-20-39-38',
    # load_model      = '2023-08-30-20-10-11', 
    robot_file      = 'quadrupbot_env//quadrupv1.urdf',
    num_robot       = 2,
    learning_rate   = 0,
    data_size       = 500,
    batch_size      = 500*2,
    epochs          = 10000,
    thresh          = 2.,
    explore         = 1e-2,
    epsilon         = 0.2,
    log_data        = False,
    save_model      = False,
    render_mode     = True, 
    norm            = False,
    print_rew       = True,
    real_time       = False,
    train_          = True,
    zeta            = 0.02,
    terrain_height  = [0., 0.0],
    debug           = True,
    run             = 1    
)
eval.train()
