import SAC
import quad_multidirect_env as qa

trainer = SAC.SAC_quad(
    envi            = qa,
    load_model      = '2023-10-28-20-52-42',
    num_robot       = 1,
    learning_rate   = 0,
    data_size       = 2000,
    batch_size      = 500,
    epochs          = 10000,
    thresh          = 10.,
    zeta            = 0.5,
    log_data        = False,
    save_model      = False,
    render_mode     = "human",
    debug           = True,
    terrain_height  = [0., 0.05],
    run             = 0,
    print_rew       = True,
    real_time       = False,
    train_          = True,
    temp            = 9999,
)
trainer.train()