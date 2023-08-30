import PPO
import quad_environment as qa

eval = PPO.PPO_quad(
    load_model      = '2023-08-30-09-04-38',
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
    real_time       = True,
    train_          = True,
    zeta            = 0.002,
    terrain_height  = [0., 0.05],
    debug           = True,
    # run             = 1    
)
eval.train()