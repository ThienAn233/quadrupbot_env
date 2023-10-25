import SAC
import quad_cnn_env_no_contact as qa

eval = SAC.SAC_quad(
    # load_model      = '2023-10-23-15-49-17_best_191.75',     
    envi            = qa,
    num_robot       = 3,
    learning_rate   = 0,
    data_size       = 10,#500,
    batch_size      = 20,#500*2,
    epochs          = 10000,
    thresh          = 2.,
    explore         = 1e-2,
    log_data        = False,
    save_model      = False,
    render_mode     = True, 
    norm            = False,
    print_rew       = True,
    real_time       = False,
    train_          = True,
    zeta            = 0.05,
    terrain_height  = [0., 0.0  ],
    debug           = False,
    run             = 1    
)
eval.train()