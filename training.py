import os
import PPO
import quad_environment as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

# Run training code
trainer = PPO.PPO_quad(
    load_model      = '2023-08-30-20-10-11', 
    robot_file      = 'quadrupbot_env\quadrupv1.urdf',
    num_robot       = 18,
    learning_rate   = 1e-4,
    data_size       = 10000,
    batch_size      = 2000,
    epochs          = 100,
    thresh          = 2.,
    explore         = 1e-2,
    epsilon         = 0.2,
    zeta            = 0.2,
    log_data        = True,
    save_model      = True,
    render_mode     = False, 
    run             = 1,    
)
trainer.train()