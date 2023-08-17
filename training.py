import PPO
import quad_environment as qa
trainer = PPO.PPO_quad(
    num_robot       = 9,
    learning_rate   =1e-4,
    data_size       =10000,
    batch_size      =2000,
    epochs          = 100,
    thresh          = 2.,
    explore         = 1e-2,
    epsilon         = 0.2,
    log_data        = True,
    save_model      = True,
    render_mode     = False, 
)
trainer.train()