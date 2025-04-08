import torch
import torch.nn as nn
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

class MambaResidualBlock(nn.Module):
    def __init__(self, config, idx=None):
        '''Residual Block wrapping Mamba, LN and Res connection'''
        super(MambaResidualBlock, self).__init__()
        
        self.cfg = config
        self.mamba = Mamba(d_model=config['d_model'],
                           d_state=config['d_state'],
                           expand=config['expand'],
                           dt_rank=config['dt_rank'],
                           d_conv=config['d_conv'])

        self.norm = RMSNorm(config['d_model'])
        self.block_mode = config['block_mode']
        self.idx = idx # block idx

    def forward(self, x, residual=None):
        '''
        Standard ResidualBlock Operation: LN->Mamba->Add
        x : (b, l, d)
        out : (b, l, d)
        '''

        if self.block_mode == 'refined':
            assert residual is not None
            y = self.norm(x)
            y = self.mamba(y)
            out = y + residual
        
        elif self.block_mode == 'interpolated':
            assert residual is not None
            y = self.norm(x)
            y = self.mamba(y)
            out = y + residual
        
        elif (self.block_mode == 'none') or (self.block_mode == 'residual'):
            y = self.norm(x)
            y = self.mamba(y)
            out = y + x
        
        else: raise Exception('Available MambaBlock modes: [none, residual, refined, interpolated]')

        return out
    
    def step(self, x, hidden_states, residual=None):
        if hidden_states['ssm_state'] is None:
            b, l, d = x.shape
            ssm_state = torch.zeros(b,
                                    self.cfg['d_model'] * self.cfg['expand'],
                                    self.cfg['d_state'],
                                    device=self.mamba.dt_proj.weight.device,
                                    dtype=self.mamba.dt_proj.weight.dtype,)

            conv_state = torch.zeros(b,
                                    self.cfg['d_model'] * self.cfg['expand'],
                                    self.cfg['d_conv'],
                                    device=self.mamba.conv1d.weight.device,
                                    dtype=self.mamba.conv1d.weight.dtype,)
        
        else:
            ssm_state, conv_state = hidden_states['ssm_state'], hidden_states['conv_state']
        
        if self.block_mode == 'refined':
            assert residual is not None
            y = self.norm(x)
            y, conv_state, ssm_state = self.mamba.step(y, conv_state, ssm_state)
            out = y + residual

        elif self.block_mode == 'interpolated':
            assert residual is not None
            y = self.norm(x)
            y, conv_state, ssm_state = self.mamba.step(y, conv_state, ssm_state)
            out = y + residual
        
        elif (self.block_mode == 'none') or (self.block_mode == 'residual'):
            y = self.norm(x)
            y, conv_state, ssm_state = self.mamba.step(y, conv_state, ssm_state)
            out = y + x
        
        else: raise Exception('Available MambaBlock modes: [none, residual, refined, interpolated]')

        return out, conv_state, ssm_state