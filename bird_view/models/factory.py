def get_model(backbone, n_step=5, ss_loss=True):
    """
    R returns the model

    Args:
        backbone: (todo): write your description
        n_step: (int): write your description
        ss_loss: (todo): write your description
    """
    if ss_loss:
        return BirdViewPolicyModelSS(backbone, n_step=n_step)
    else:
        return BirdViewPolicyModel(backbone, n_step=n_step)

