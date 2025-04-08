import torch

def pred_motion(model, inputs):
    if model == 'CV':
       CV = torch.Tensor([[1, 0, 0, 0, 0, 0, 0.5, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0.5, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]]).to(inputs.device)
       
       preds = inputs@CV.T
    
    return preds