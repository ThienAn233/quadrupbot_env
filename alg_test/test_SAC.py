import os
import test_env as te
import SAC

# Open tensor board
os.popen('tensorboard --logdir=quadrupbot_env\\alg_test\\runs')

trainer = SAC.SAC_quad(
    envi            = te,
    PATH            = "quadrupbot_env//alg_test//",
    num_robot       = 1,
    learning_rate   = 1e-2,
    data_size       = 1000,
    batch_size      = 1000,
    epochs          = 100,
    thresh          = -10.,
    zeta            = 0.5,
    log_data        = True,
    save_model      = True,
    render_mode     = 'human',
    run             = 0,
    temp            = 1,
)
trainer.train()