import torch
import torch.nn as nn
from model.modules import MambaResidualBlock
from mamba_ssm.ops.triton.layer_norm import RMSNorm
import numpy as np

class MambaMotionPrediction(nn.Module):
    def __init__(self, config):
        super(MambaMotionPrediction, self).__init__()
        
        dim = int(config['intermediate_features'])
        dim_list = [dim//4, dim//2]

        self.mode = config['mode']
        self.block_mode = config['block_mode'] # how cv context is mixed with the model

        self.input_mlp = nn.Sequential(
            nn.Linear(config['input_features'], dim_list[0]),
            nn.ReLU(),
            nn.Linear(dim_list[0], dim_list[1]),
            nn.ReLU(),
            nn.Linear(dim_list[1], dim),
        )

        self.layers = nn.ModuleList(MambaResidualBlock(config=config['mamba'], idx=i) for i in range(config['depth']))
        self.norm = RMSNorm(dim)

        self.output_mlp = nn.Sequential(
            nn.Linear(dim, dim_list[1]),
            nn.ReLU(),
            nn.Linear(dim_list[1], dim_list[0]),
            nn.ReLU(),
            nn.Linear(dim_list[0], config['output_features']),
        )

        self.hidden_states = {idx:{'conv_state': None,
                                   'ssm_state': None} for idx in range(len(self.layers))}
        
        self.dt = 0.5
        self.device = config['device']
        self.CV = self.getCV(dt=self.dt)
        self.interCV = self.getCV(dt=(self.dt/config['depth'])) # cv tf matrix dt = (dt/nBlocks)

    def getCV(self, dt:float):
        cv = torch.Tensor([[1, 0, 0, 0, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]]).to(self.device)
        return cv

    def forward(self, x):
        if self.mode == 'sequence':
            return self.sequence(x)
        elif self.mode == 'mix':
            return self.mix(x, pred_length=6)
        else: raise Exception('Supported Training modes: [sequence, mix]')

    def sequence(self, x):
        '''
        x: (b, l, d) --> (b, l, d_inter) --> (b, l, d)
        seq: [T1, T2, .., Tn] --> [T2, T3, .., Tn+1]
        '''

        # mamba model with no cv context
        if self.block_mode == 'none':
            x = self.input_mlp(x)

            for layer in self.layers:
                x = layer(x)
            
            x = self.norm(x)
            x = self.output_mlp(x)

            return x

        # mamba model used to predict cv offset
        elif self.block_mode == 'residual':
            cv = x@self.CV.T
            x = self.input_mlp(x)

            for layer in self.layers:
                x = layer(x)
            
            x = self.norm(x)
            x = self.output_mlp(x)
            x = x + cv

            return x
        
        # in refined mode, we add cv (full dt) context as residual connection in the first block only, the following blocks refine the predicitons.
        elif self.block_mode == 'refined':
            cv = x@self.CV.T
            x = self.input_mlp(x)

            for layer in self.layers:
                if layer.idx == 0:
                    res = self.input_mlp(cv)
                    x = layer(x, residual=res)
                else:
                    x = layer(x, residual=x)
            
            x = self.norm(x)
            x = self.output_mlp(x)

            return x
        
        # each block predicts motion of dt/nBlocks timestep, and it's corresponding cv_preds are added as residual 
        elif self.block_mode == 'interpolated':
            cv = x@self.CV.T
            res = x@self.interCV.T
            x = self.input_mlp(x)

            for layer in self.layers:
                if layer.idx == 0:
                    res = self.input_mlp(res)
                    x = layer(x, residual=res)
                else:
                    res = self.output_mlp(x)
                    res = res@self.interCV.T
                    res = self.input_mlp(res)
                    x = layer(x, residual=res)
            
            x = self.norm(x)
            x = self.output_mlp(x)

            return x

        else: raise Exception('Available MambaBlock modes: [none, residual, refined, interpolated]')

    def step(self, x):
        '''
        x: (b, l, d) --> (b, l, d_inter) --> (b, l, d)
        seq: Tn -> Tn+1
        '''

        # mamba model with no cv context
        if self.block_mode == 'none':
            x = self.input_mlp(x)

            for layer_idx, layer in enumerate(self.layers):
                x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx])

                # cache hidden states
                self.hidden_states[layer_idx].update({'conv_state': conv_state,
                                                  'ssm_state': ssm_state})

            
            x = self.norm(x)
            x = self.output_mlp(x)

            return x

        # mamba model used to predict cv offset
        elif self.block_mode == 'residual':
            cv = x@self.CV.T
            x = self.input_mlp(x)

            for layer_idx, layer in enumerate(self.layers):
                x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx])

                # cache hidden states
                self.hidden_states[layer_idx].update({'conv_state': conv_state,
                                                  'ssm_state': ssm_state})
            
            x = self.norm(x)
            x = self.output_mlp(x)
            x = x + cv

            return x
        
        # in refined mode, we add cv (full dt) context as residual connection in the first block only, the following blocks refine the predicitons.
        elif self.block_mode == 'refined':
            cv = x@self.CV.T
            x = self.input_mlp(x)

            for layer_idx, layer in enumerate(self.layers):
                if layer.idx == 0:
                    res = self.input_mlp(cv)
                    x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx], residual=res)
                else:
                    x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx], residual=x)

                # cache hidden states
                self.hidden_states[layer_idx].update({'conv_state': conv_state,
                                                  'ssm_state': ssm_state})
                     
            x = self.norm(x)
            x = self.output_mlp(x)

            return x
        
        # each block predicts motion of dt/nBlocks timestep, and it's corresponding cv_preds are added as residual 
        elif self.block_mode == 'interpolated':
            cv = x@self.CV.T
            res = x@self.interCV.T
            x = self.input_mlp(x)

            for layer_idx, layer in enumerate(self.layers):
                if layer.idx == 0:
                    res = self.input_mlp(res)
                    x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx], residual=res)
                else:
                    res = self.output_mlp(x)
                    res = res@self.interCV.T
                    res = self.input_mlp(res)
                    x, conv_state, ssm_state = layer.step(x, self.hidden_states[layer_idx], residual=res)

                # cache hidden states
                self.hidden_states[layer_idx].update({'conv_state': conv_state,
                                                  'ssm_state': ssm_state})
            
            x = self.norm(x)
            x = self.output_mlp(x)

            return x
        
        else: raise Exception('Available MambaBlock modes: [none, residual, refined, interpolated]')
  
    def mix(self, x, pred_length=6):
        '''
        x: (b, l, d) --> (b, l, d_inter) --> (b, l, d)
        seq: [T1, T2, .., Tn] --> [Tn+1, ..., Tn+pred_len]
        '''
        # reset states (mix is only used at training, so reset the states for every new sequence)
        self.hidden_states = {idx:{'conv_state': None,
                            'ssm_state': None} for idx in range(len(self.layers))}
    
        b, l, d = x.shape
        for i in range(l):
            y = self.step(x[:,i:i+1,:])
        
        # y is the predicted Tn+1
        outs = [y]
        # auto regressively predict further steps
        for _ in range(pred_length-1):
            y = self.step(y)
            outs.append(y)

        outs = torch.stack(outs, dim=1).squeeze(2)

        return outs
    
    @torch.no_grad
    def predict(self, state):
        '''
        pure inference, hidden states are never manually reset
        state: expected to be dtype float32 either as np.ndarray or torch.Tensor of dim(9)
        pred: np.mat of dtype float32
        '''
        # convert to Tensor
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            x = state.to(self.device)
        else:
            raise Exception('State expected to be dtype float32 either as np.ndarray or torch.Tensor of dim(9)')
        
        x = x.reshape(1,1,-1) # b l d
        x = self.step(x)
        x = x.cpu().numpy().reshape(-1)
        x = np.mat(x).T
        return x
    
    @torch.no_grad
    def predictEgoCentric(self, state):
        '''
        inference, use n past trajectories to predict current trajectory
        hidden states are reset for every mini sequence
        state: expected to be dtype np.ndarray (l,dim) in float32
        pred: np.mat of dtype float32
        '''
        x = torch.from_numpy(state).to(self.device).unsqueeze(0) # b l d
        x = self.sequence(x)
        x = x[:,-1,:]
        x = x.cpu().numpy().reshape(-1)
        x = np.mat(x).T
        return x
    
    @torch.no_grad
    def predictEgoCentricUnscented(self, state):
        '''
        inference, use n past trajectories to predict current trajectory
        hidden states are reset for every mini sequence
        state: expected to be dtype np.ndarray (l,dim) in float32
        pred: np.mat of dtype float32
        '''
        x = state.to(self.device)
        x = self.sequence(x)
        return x
    
    @staticmethod
    def warp_to_pi(yaw: float) -> float:
        """warp yaw to [-pi, pi)

        Args:
            yaw (float): raw angle

        Returns:
            float: raw angle after warping
        """
        while yaw >= np.pi:
            yaw = yaw - 2 * np.pi
        while yaw < -np.pi:
            yaw = yaw + 2 * np.pi
        return yaw
    
    @staticmethod
    def fullwrap(yaws):

        yaws = yaws % 360
        yaws = (yaws + 360) % 360
        return torch.where(yaws > 180, yaws-360, yaws)