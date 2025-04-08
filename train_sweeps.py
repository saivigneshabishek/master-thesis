# run me with environment=environment +experiment=train_mp
import os
import torch
import wandb
import hydra
from tqdm import tqdm
from dataloader import create_dataloader
from model import create_model
from loss import create_loss
from omegaconf import DictConfig, OmegaConf
import numpy as np
from einops import rearrange
from train_utils import pred_motion

def train(dataloader, model, loss_fn, optimizer, scheduler, wandb):
    model.train()
    total_loss = 0.0
    for inputs, targets, mask in tqdm(dataloader):
        inputs, targets, mask = inputs.to('cuda').float(), targets.to('cuda').float(), mask.to('cuda').bool()

        # first approach treat multi agents as batches
        inputs = rearrange(inputs, 'b l n d -> (b n) l d')
        targets = rearrange(targets, 'b l n d -> (b n) l d')
        mask = rearrange(mask, 'b l n d -> (b n) l d')

        output = model(inputs)
        loss = loss_fn(output, targets, mask)
        total_loss += loss.item()
        wandb.log({"train/loss": loss.item(),
                   "train/lr": scheduler.get_last_lr()[0]})
        
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"train/avg_loss": avg_loss})

    return avg_loss

@torch.no_grad
def eval(dataloader, model, loss_fn, metrics, wandb):
    model.eval()
    print('Running Evaluation data')
    eval_loss = 0.0
    ade_loss = []
    fde_loss = []
    cv_ade_loss = []
    cv_fde_loss = []
    cv_eval_loss = 0.0

    for inputs, targets, mask in tqdm(dataloader):
        inputs, targets, mask = inputs.to('cuda').float(), targets.to('cuda').float(), mask.to('cuda').bool()

        inputs = rearrange(inputs, 'b l n d -> (b n) l d')
        targets = rearrange(targets, 'b l n d -> (b n) l d')
        mask = rearrange(mask, 'b l n d -> (b n) l d')

        # perform eval metrics on preds from classical motion models
        new_preds = pred_motion(model='CV', inputs=inputs)
        cv_loss = loss_fn(new_preds, targets, mask)
        cv_ade, cv_fde = metrics(new_preds, targets, mask)
        cv_ade_loss.extend(cv_ade)
        cv_fde_loss.extend(cv_fde)
        cv_eval_loss += cv_loss.item()
        
        output = model(inputs)
        loss = loss_fn(output, targets, mask)
        ade, fde = metrics(output, targets, mask)

        ade_loss.extend(ade)
        fde_loss.extend(fde)
        eval_loss += loss.item()

    eval_avg = eval_loss / len(dataloader)
    ade_avg = torch.mean(torch.Tensor(ade_loss), dim=0).item()
    fde_avg = torch.mean(torch.Tensor(fde_loss), dim=0).item()

    CV_ADE = torch.mean(torch.Tensor(cv_ade_loss), dim=0).item()
    CV_FDE = torch.mean(torch.Tensor(cv_fde_loss), dim=0).item()
    CV_LOSS = cv_eval_loss/len(dataloader)

    wandb.log({"val/avg_loss": eval_avg,
               "val/ade_loss": ade_avg,
               "val/fde_loss": fde_avg,
               "val/cv_ade": CV_ADE,
               "val/cv_fde": CV_FDE,
               "val/cv_loss": CV_LOSS,
               })

    return eval_avg

@hydra.main(config_path="config", config_name="train_config", version_base="1.3")
def main(cfg):

    cfg = OmegaConf.to_object(cfg)
    # setup directories
    dir = os.path.join(cfg['environment']['output_base_path'], 'sweeps', 'INTERPOLATED_LR', 'checkpoints', cfg["logging"]["name"])
    run = wandb.init(dir=dir, config=cfg)
    ckpt_path = os.path.join(dir, run.name)
    os.makedirs(dir, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
 
    # use sweep values
    cfg['training']['learning_rate'] = wandb.config.lr
    # cfg['model']['intermediate_features'] = wandb.config.dim
    # cfg['model']['mamba']['d_model'] = wandb.config.dim
    # cfg['model']['depth'] = wandb.config.depth
    # cfg['model']['mamba']['expand'] = wandb.config.expand
    # cfg['model']['mamba']['d_state'] = wandb.config.d_state
    # cfg['model']['mamba']['dt_rank'] = wandb.config.dt_rank
    # cfg['model']['mamba']['d_conv'] = wandb.config.d_conv

    # wandb.config = cfg

    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    # initialize dataloaders, model, loss, optimizer
    train_dataloader = create_dataloader(cfg['dataloader'], isTrain=True)
    eval_dataloader = create_dataloader(cfg['dataloader'], isTrain=False)
    
    model = create_model(cfg['model'])
    loss_fn = create_loss(cfg['loss'])
    metrics = create_loss(cfg['metrics'])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    num_epochs = cfg['training']['epochs']
    best_loss = float('inf')
    tr_best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=cfg['training']['learning_rate'],
                                                    epochs=num_epochs,
                                                    steps_per_epoch=len(train_dataloader))

    for epoch in range(num_epochs):
        # training loop
        avg_loss = train(train_dataloader, model, loss_fn, optimizer, scheduler, wandb)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss}")
        train_dataloader.dataset.cache = True

        if avg_loss < tr_best_loss:
            tr_best_loss = avg_loss

        # do evaluation every few epochs
        if (epoch % 2 == 0) or (epoch == num_epochs-1):
            eval_avg = eval(eval_dataloader, model, loss_fn, metrics, wandb)
            print(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {eval_avg}")
            eval_dataloader.dataset.cache = True

            # save best model
            if eval_avg < best_loss:
                print('Saving Best Model')
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, 'model_best.pth'))
                best_loss = eval_avg

        if (epoch==num_epochs-1):
            print('Saving Last Model')
            checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, os.path.join(ckpt_path, 'model_last.pth'))
    
    wandb.log({'train/best_loss': tr_best_loss})
    wandb.finish()

if __name__ == "__main__":
    main()
