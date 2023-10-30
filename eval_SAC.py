import SAC
import quad_multidirect_env as qa

trainer = SAC.SAC_quad(
    envi            = qa,
    load_model      = '2023-10-29-16-41-24_best_64.19',
    num_robot       = 1,
    learning_rate   = 0,
    data_size       = 500,
    batch_size      = 500,
    epochs          = 200,
    thresh          = 2.,
    zeta            = .5,
    terrain_height  = [0, 0.],
    log_data        = False,
    save_model      = False,
    render_mode     = 'human',
    debug           = True,
    temp            = 9999999,
    run             = 1,
)
trainer.train()