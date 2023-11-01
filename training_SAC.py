import os
import SAC
import quad_cnn_env_no_contact as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

trainer = SAC.SAC_quad(
    envi            = qa,
    num_robot       = 5,
    learning_rate   = 1e-4,
    data_size       = 1000,
    batch_size      = 2500-1,
    epochs          = 200,
    thresh          = 30.,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = None,
    debug           = False,
    terrain_height  = [0., 0.0],
    run             = 1,
    temp            = 0.96,
)
trainer.train()