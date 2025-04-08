import dataloader.dataloader as all_datasets
from torch.utils.data import DataLoader


def create_dataloader(cfg, isTrain=None):
    if isTrain is True:
        dataset = getattr(all_datasets, cfg['name'])(cfg, isTrain=isTrain)
        dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers']) # etc....
    elif isTrain is False:
        dataset = getattr(all_datasets, cfg['name'])(cfg, isTrain=isTrain)
        dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers']) # etc....
    else:
        dataset = getattr(all_datasets, cfg['name'])(cfg)
        dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers']) # etc....

    return dataloader