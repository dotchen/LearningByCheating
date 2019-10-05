def get_model(backbone, n_step=5, ss_loss=True):
    if ss_loss:
        return BirdViewPolicyModelSS(backbone, n_step=n_step)
    else:
        return BirdViewPolicyModel(backbone, n_step=n_step)

