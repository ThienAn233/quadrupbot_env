import SAC
import quad_cnn_env_no_contact as qa

eval = SAC.SAC_quad(
    load_model      = '2023-10-18-16-41-34_best_93.52',     
    envi            = qa,
    num_robot       = 2,
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
    zeta            = 0.5,
    terrain_height  = [0., 0.05],
    debug           = True,
    run             = 1    
)
eval.train()