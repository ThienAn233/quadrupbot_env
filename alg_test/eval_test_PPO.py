import test_env as te
import PPO
import os

# Run training code
trainer = PPO.PPO_quad(
    envi            = te,
    PATH            = "quadrupbot_env//alg_test//",
    load_model      = '2023-10-27-19-29-03_best_15.43',
    num_robot       = 1,
    learning_rate   = 0,
    data_size       = 100000,
    batch_size      = 1000,
    epochs          = 100,
    thresh          = 2.,
    explore         = 1e-4,
    epsilon         = 0.1,
    zeta            = .5,
    log_data        = False,
    save_model      = False,
    render_mode     = "human", 
    norm            = False,
    run             = 0,   
    minmax          = [-3.,3.],
    action_space    = 1,
    observation_space= 3, 
)
trainer.train()