import PPO_cnn
import quad_cnn_env_no_contact as qa

eval = PPO_cnn.PPO_quad(
    load_model      = '2023-09-27-16-58-18',     
    envi            = qa,
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
    zeta            = 0.05,
    terrain_height  = [0., 0.0],
    debug           = True,
    run             = 1    
)
eval.train()