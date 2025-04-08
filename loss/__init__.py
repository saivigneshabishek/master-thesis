import loss.loss as all_loss

# create a loss function based on the config
def create_loss(cfg):
    loss_func = getattr(all_loss, cfg["name"])()
    return loss_func