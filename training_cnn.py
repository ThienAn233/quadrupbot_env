import os
import PPO_cnn
import quad_cnn_env_no_contact as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

# Run training code
trainer = PPO_cnn.PPO_quad(
    # load_model      = '2023-09-27-16-58-18',
    envi            = qa,
    norm            = True,
    num_robot       = 9,
    learning_rate   = 5e-4,
    data_size       = 10000,
    batch_size      = 2000,
    epochs          = 10,
    thresh          = 2.,
    explore         = 1e-2,
    epsilon         = 0.2,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = False, 
    run             = 1,    
)
trainer.train()