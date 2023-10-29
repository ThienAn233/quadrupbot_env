import os
import SAC
import quad_multidirect_env as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

trainer = SAC.SAC_quad(
    envi            = qa,
    num_robot       = 5,
    learning_rate   = 3e-3,
    data_size       = 500,
    batch_size      = 2500,
    epochs          = 200,
    thresh          = 300.,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = None,
    debug           = False,
    terrain_height  = [0., 0.0],
    run             = 1,
    temp            = 1.09,
)
trainer.train()