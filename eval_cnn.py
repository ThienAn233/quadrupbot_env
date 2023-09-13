import PPO_cnn
import quad_cnn_env as qa

eval = PPO_cnn.PPO_quad(
    load_model      = '2023-09-12-18-50-57', 
    num_robot       = 3,
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
    zeta            = 0.5,
    terrain_height  = [0., 0.05],
    debug           = True,
    run             = 1    
)
eval.train()