import SAC
import quad_cnn_env_no_contact as qa

trainer = SAC.SAC_quad(
    envi            = qa,
    load_model      = '2023-11-01-03-12-26',
    num_robot       = 1,
    learning_rate   = 0,
    data_size       = 500,
    batch_size      = 500,
    epochs          = 200,
    thresh          = 2.,
    zeta            = .05,
    terrain_height  = [0, 0.],
    log_data        = False,
    save_model      = False,
    real_time       = True,
    render_mode     = 'human',
    debug           = True,
    temp            = 9999999,
    run             = 1,
)
trainer.train()