import test_env as te
import alg_test_PPO
import os

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\alg_test\\runs')

# Run training code
trainer = alg_test_PPO.PPO_quad(
    envi            = te,
    PATH            = "quadrupbot_env//alg_test//",
    num_robot       = 1,
    learning_rate   = 1e-3,
    data_size       = 1000,
    batch_size      = 200,
    gamma           = 1.,
    epochs          = 100,
    thresh          = 20.,
    explore         = 1e-1,
    epsilon         = 0.1,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = None, 
    print_rew       = False,
    run             = 0,   
    norm            = False,
    minmax          = [-1.,1.],
    action_space    = 4,
    observation_space= 24, 
)
trainer.train()