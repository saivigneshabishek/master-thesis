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
import matplotlib.pyplot as plt

def tonumpy(items):
    ret = [item.cpu().numpy() for item in items]
    return ret

def annotate_boxplot(ax, box_data, position):
    med = box_data['medians'][0].get_ydata()[0]
    q3 = box_data['boxes'][0].get_ydata()[2]
    q1 = box_data['boxes'][0].get_ydata()[0]

    # Annotate median
    ax.annotate(f'Median: {med:.3f}', xy=(position, med), xytext=(position - 0.19, med * 1.05),
                fontsize=8, color='blue')
    
    # Annotate upper quartile (Q3)
    ax.annotate(f'Q3: {q3:.3f}', xy=(position, q3), xytext=(position - 0.19, q3 * 1.05),
                fontsize=8, color='green')
    
    # Annotate lower quartile (Q1)
    ax.annotate(f'Q1: {q1:.3f}', xy=(position, q1), xytext=(position - 0.19, q1 * 1.05),
                fontsize=8, color='red')

    # Annotate the topmost outlier if it exists
    outliers = box_data['fliers'][0].get_ydata()
    if len(outliers) > 0:
        top_outlier = max(outliers)
        ax.annotate(f'Top Outlier: {top_outlier:.3f}', xy=(position, top_outlier),
                    xytext=(position + 0.19, top_outlier),
                    arrowprops=dict(arrowstyle='->'), fontsize=8, color='purple')

def plot_graphs(data_cv, data_model, path, type='ade'):
    assert type in ['ade', 'fde']

    fig, ax = plt.subplots(figsize = (9, 6))
    # Creating plot
    cv = plt.boxplot(data_cv, positions=[1], widths=0.4)
    model = plt.boxplot(data_model, positions=[2], widths=0.4)

    ax.set_xticks([1,2])
    ax.set_xticklabels(['CV', 'Model'])
    ax.set_yscale("log")

    annotate_boxplot(ax, cv, 1)
    annotate_boxplot(ax, model, 2)

    if type == 'ade':
        plt.savefig(os.path.join(path, 'ADE.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path, 'FDE.png'), dpi=300, bbox_inches='tight')
    plt.close()


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

    return eval_avg, ade_loss, fde_loss, cv_ade_loss, cv_fde_loss

@hydra.main(config_path="config", config_name="train_config", version_base="1.3")
def main(cfg):

    # setup directories
    graph_path = os.path.join(cfg['environment']['output_base_path'], 'graphruns', cfg["logging"]["name"])
    os.makedirs(graph_path, exist_ok=True)

    OmegaConf.to_object(cfg)
    
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)
    # initialize dataloaders, model, loss, optimizer
    eval_dataloader = create_dataloader(cfg['dataloader'], isTrain=False)
    
    model = create_model(cfg['model'])
    ckpt = torch.load('/home/sai/2024-mt-deenadayalan/final_outputs/checkpoints/MAMBA_RESIDUAL_LR_0_0007952/model_best.pth')
    # ckpt = torch.load('/home/sai/2024-mt-deenadayalan/final_outputs/sweeps/BASE_LR/checkpoints/MAMBA_BASE_LR/major-sweep-4/model_best.pth')
    ckpt = ckpt['model'] if 'model' in ckpt.keys() else ckpt
    model.load_state_dict(ckpt)
    loss_fn = create_loss(cfg['loss'])
    metrics = create_loss(cfg['metrics'])
    num_epochs = 1

    # do evaluation
    for epoch in range(num_epochs):
        eval_avg, ade_loss, fde_loss, cv_ade_loss, cv_fde_loss = eval(eval_dataloader, model, loss_fn, metrics, wandb)
        print(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {eval_avg}")
        eval_dataloader.dataset.cache = True

        # save the ades and fdes
        torch.save(ade_loss, os.path.join(graph_path, 'ade.pt'))
        torch.save(cv_ade_loss,  os.path.join(graph_path, 'cv_ade.pt'))
        torch.save(fde_loss,  os.path.join(graph_path, 'fde.pt'))
        torch.save(cv_fde_loss,  os.path.join(graph_path, 'cv_fde.pt'))

        # plot ade boxplot
        ade_loss = tonumpy(ade_loss)
        cv_ade_loss = tonumpy(cv_ade_loss)
        plot_graphs(cv_ade_loss, ade_loss, graph_path, type='ade')

        # plot fde boxplot
        fde_loss = tonumpy(fde_loss)
        cv_fde_loss = tonumpy(cv_fde_loss)
        plot_graphs(cv_fde_loss, fde_loss, graph_path, type='fde')

if __name__ == "__main__":
    main()
