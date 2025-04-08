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

def annotate_boxplot(ax, box_data, data, position):
    med = box_data['medians'][0].get_ydata()[0]
    q3 = box_data['boxes'][0].get_ydata()[2]
    q1 = box_data['boxes'][0].get_ydata()[0]
    mean = np.mean(data)

    # Annotate median
    ax.annotate(f'Median: {med:.3f}', xy=(position, med), xytext=(position - 0.19, med * 1.05),
                fontsize=8, color='blue')
    
    # Annotate upper quartile (Q3)
    ax.annotate(f'Q3: {q3:.3f}', xy=(position, q3), xytext=(position - 0.19, q3 * 1.05),
                fontsize=8, color='green')
    
    # Annotate lower quartile (Q1)
    ax.annotate(f'Q1: {q1:.3f}', xy=(position, q1), xytext=(position - 0.19, q1 * 1.05),
                fontsize=8, color='red')
    
    # Annotate mean
    ax.annotate(f'Mean: {mean:.3f}', xy=(position, mean), xytext=(position + 0.21, mean * 0.95),
                fontsize=8, color='magenta')
    
    # Draw mean line
    ax.hlines(mean, position - 0.2, position + 0.2, colors='magenta', linestyles='dotted', label='Mean')
    
    # Annotate the topmost outlier if it exists
    outliers = box_data['fliers'][0].get_ydata()
    if len(outliers) > 0:
        top_outlier = max(outliers)
        ax.annotate(f'Top Outlier: {top_outlier:.3f}', xy=(position, top_outlier),
                    xytext=(position + 0.19, top_outlier),
                    arrowprops=dict(arrowstyle='->'), fontsize=8, color='purple')

def plot_graphs(data_cv, data_modelBase, data_modelA, data_modelB, data_modelC, path, type='ADE'):
    assert type in ['ADE', 'FDE']

    fig, ax = plt.subplots(figsize = (20, 12))
    # Creating plot
    cv = plt.boxplot(data_cv, positions=[1], widths=0.4)
    modelBase = plt.boxplot(data_modelBase, positions=[2], widths=0.4)
    modelA = plt.boxplot(data_modelA, positions=[3], widths=0.4)
    modelB = plt.boxplot(data_modelB, positions=[4], widths=0.4)
    modelC = plt.boxplot(data_modelC, positions=[5], widths=0.4)

    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels(['CV', 'Baseline Mamba', 'Residual Mamba', 'Refined Mamba', 'Interpolated Mamba'])
    ax.set_yscale("log")
    namee = 'Average Displacement Error' if type=='ADE' else 'Final Displacement Error'
    ax.set_title(f'{namee}')

    annotate_boxplot(ax, cv, data_cv, 1)
    annotate_boxplot(ax, modelBase, data_modelBase, 2)
    annotate_boxplot(ax, modelA, data_modelA, 3)
    annotate_boxplot(ax, modelB, data_modelB, 4)
    annotate_boxplot(ax, modelC, data_modelC, 5)


    if type == 'ADE':
        plt.savefig(os.path.join(path, 'ADE.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path, 'FDE.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():

    mainpath = '/home/sai/2024-mt-deenadayalan/final_outputs/graphruns/FINAL/SEQ_5/'
    folders = ['MAMBA_BASE_LR_0_001815', 'MAMBA_RESIDUAL_LR_0_0007952', 'MAMBA_REFINED_LR_0_00242', 'MAMBA_INTERPOLATED_LR_0_00219']


    # load the ades
    cv_ade_loss = torch.load(os.path.join(mainpath, folders[0], 'cv_ade.pt'))
    base_ade_loss = torch.load(os.path.join(mainpath, folders[0], 'ade.pt'))
    A_ade_loss = torch.load(os.path.join(mainpath, folders[1], 'ade.pt'))
    B_ade_loss = torch.load(os.path.join(mainpath, folders[2], 'ade.pt'))
    C_ade_loss = torch.load(os.path.join(mainpath, folders[3], 'ade.pt'))

    # plot ade boxplot
    base_ade_loss = tonumpy(base_ade_loss)
    cv_ade_loss = tonumpy(cv_ade_loss)
    A_ade_loss = tonumpy(A_ade_loss)
    B_ade_loss = tonumpy(B_ade_loss)
    C_ade_loss = tonumpy(C_ade_loss)

    plot_graphs(cv_ade_loss, base_ade_loss, A_ade_loss, B_ade_loss, C_ade_loss, mainpath, type='ADE')

    # load the fdes
    cv_ade_loss = torch.load(os.path.join(mainpath, folders[0], 'cv_fde.pt'))
    base_ade_loss = torch.load(os.path.join(mainpath, folders[0], 'fde.pt'))
    A_ade_loss = torch.load(os.path.join(mainpath, folders[1], 'fde.pt'))
    B_ade_loss = torch.load(os.path.join(mainpath, folders[2], 'fde.pt'))
    C_ade_loss = torch.load(os.path.join(mainpath, folders[3], 'fde.pt'))

    # plot ade boxplot
    base_ade_loss = tonumpy(base_ade_loss)
    cv_ade_loss = tonumpy(cv_ade_loss)
    A_ade_loss = tonumpy(A_ade_loss)
    B_ade_loss = tonumpy(B_ade_loss)
    C_ade_loss = tonumpy(C_ade_loss)

    plot_graphs(cv_ade_loss, base_ade_loss, A_ade_loss, B_ade_loss, C_ade_loss, mainpath, type='FDE')

if __name__ == "__main__":
    main()
