import os
import SAC
import quad_multidirect_env as qa

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\runs')

trainer = SAC.SAC_quad(
    envi            = qa,
    num_robot       = 9,
    learning_rate   = 5e-4,
    data_size       = 90000,
    batch_size      = 2000,
    epochs          = 100,
    thresh          = 2.,
    explore         = 1e-2,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = None,
    debug           = False,
    run             = 1,
)
trainer.train()