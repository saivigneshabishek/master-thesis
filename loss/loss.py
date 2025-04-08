import torch
import torch.nn as nn

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.loss_fn = nn.L1Loss(reduction='none')
    
    def forward(self, preds, targets, mask):
        masked_loss = self.loss_fn(preds, targets) * mask
        loss = torch.sum(masked_loss)/torch.sum(mask)
        return loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, preds, targets, mask):
        masked_loss = self.loss_fn(preds, targets) * mask
        loss = torch.sum(masked_loss)/torch.sum(mask)
        return loss
    
class EvalMetrics(nn.Module):
    '''used as an eval metric'''
    def __init__(self):
        super(EvalMetrics, self).__init__()

    @torch.no_grad
    def forward(self, preds, targets, mask):
        ade = self.avg_ade(preds, targets, mask)
        fde = self.avg_fde(preds, targets, mask)
        return ade, fde
    
    def avg_ade(self, preds, targets, mask):
        '''sqrt((preds-targets)**2)'''
        mask = mask[:,0,0].bool()
        ade = torch.mean(torch.linalg.norm(((preds[mask][:,:,:3]-targets[mask][:,:,:3])), dim=-1), dim=-1)
        return ade
    
    def avg_fde(self, preds, targets, mask):
        '''sqrt((preds-targets)**2)'''
        mask = mask[:,0,0].bool()
        fde = torch.linalg.norm(((preds[mask][:,-1,:3]-targets[mask][:,-1,:3])), dim=-1)
        return fde