import os
import SAC
import quad_multidirect_env as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

trainer = SAC.SAC_quad(
    envi            = qa,
    num_robot       = 3,
    learning_rate   = 1e-4,
    data_size       = 500,
    batch_size      = 3*500,
    epochs          = 100,
    thresh          = 40.,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = None,
    debug           = False,
    terrain_height  = [0., 0.0],
    run             = 0,
    temp            = 1,
)
trainer.train()