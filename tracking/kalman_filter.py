import torch
from einops import rearrange
import numpy as np
import tracking.motion_models as motion_models

class KalmanFilter:
    '''Kalman Filter for CV and CA models'''
    def __init__(self, cfg, det=None, cls=0):

        dt = 1/cfg['frequency']
        model = cfg['motion_model'][cls]
        assert model in ['CV', 'CA'], 'only CV and CA models can be used with a regular KalmanFilter. For other models, use ExtendedKalmanFilter'

        if model == 'CV':
            if cfg.get('CV_Tuning') is not None:
                self.motion_model = getattr(motion_models, model)(dt, cfg['CV_Tuning'])
            else:
                self.motion_model = getattr(motion_models, model)(dt)
        else:
            self.motion_model = getattr(motion_models, model)(dt)
        self.n = self.motion_model.n
        self.state = self.motion_model.getInitState(det)

        # transition matrices
        self.F = self.motion_model.getTransitionF()
        self.Q = self.motion_model.getProcessNoiseQ()
        self.P = self.motion_model.getInitCovP()

        # measurement matrices
        self.H = self.motion_model.getMeaStateH()
        self.R = self.motion_model.getMeaNoiseR()

    def predict(self):
        # predict the state and covariance
        self.state = self.F * self.state
        self.P = self.F * self.P * self.F.T + self.Q
        self.motion_model.warpStateYawToPi(self.state)
    
    def update(self, curr_det):
        # return if no current det
        if curr_det is None:
            return
        
        # update the state with current det
        measurement = np.mat(curr_det).T
        res = measurement - (self.H * self.state)
        self.motion_model.warpResYawToPi(res)

        S = (self.H * self.P * self.H.T) + self.R
        GAIN = self.P * self.H.T * S.I

        self.state += GAIN * res
        self.P = (np.mat(np.identity(self.n)) - GAIN * self.H) * self.P
        self.motion_model.warpStateYawToPi(self.state)

    def getState(self):
        # returns [x, y, z, w, l, h, vx, vy, ry]
        return self.motion_model.getOutputInfo(self.state)
    
class ExtendedKalmanFilter():
    ''' Extended Kalman Filter for CTRA, CTRV, Bicycle models'''
    def __init__(self, cfg, det=None, cls=0):

        dt = 1/(cfg['frequency'])
        model = cfg['motion_model'][cls]
        assert model in ['CTRA', 'CTRV', 'Bicycle'], 'For other models (CV, CA), use a regular KalmanFilter'

        self.motion_model = getattr(motion_models, model)(dt)
        self.n = self.motion_model.n
        self.state = self.motion_model.getInitState(det)

        # transition matrices
        self.Q = self.motion_model.getProcessNoiseQ()
        self.P = self.motion_model.getInitCovP(cfg[model], cls)

        # measurement matrices
        self.R = self.motion_model.getMeaNoiseR()
    
    def predict(self):
        # state transition matrix
        self.F = self.motion_model.getTransitionF(self.state)
        # predict state and covariance
        self.state = self.motion_model.stateTransition(self.state)
        self.P = self.F * self.P * self.F.T + self.Q
        self.motion_model.warpStateYawToPi(self.state)

    def update(self, curr_det):
        # return if no current det
        if curr_det is None:
            return
        
        # measurement matrix
        self.H = self.motion_model.getMeaStateH(self.state)
        # update the state with current det
        measurement = np.mat(curr_det).T
        state = self.motion_model.StateToMeasure(self.state)
        res = measurement - state
        self.motion_model.warpResYawToPi(res)

        S = (self.H * self.P * self.H.T) + self.R
        GAIN = self.P * self.H.T * S.I
        I_KH = np.mat(np.identity(self.n)) - GAIN * self.H

        self.state += GAIN * res
        self.motion_model.warpStateYawToPi(self.state)
        self.P = (I_KH * self.P * I_KH.T) + (GAIN * self.R * GAIN.T)

    def getState(self):
        # returns [x, y, z, w, l, h, vx, vy, ry]
        return self.motion_model.getOutputInfo(self.state)
    
class MambaFilter:
    '''MambaFilter for SSM model'''
    def __init__(self, cfg, det=None, cls=0):

        dt = 1/cfg['frequency']
        model = cfg['motion_model'][cls]
        assert model=='SSM', 'only SSM models can be used with MambaFilter'

        self.motion_model = getattr(motion_models, model)(dt, cfg)
        self.n = self.motion_model.n
        self.state = self.motion_model.getInitState(det)

        # transition matrices
        self.F = self.motion_model.getTransitionF()
        self.Q = self.motion_model.getProcessNoiseQ()
        self.P = self.motion_model.getInitCovP()

        # measurement matrices
        self.H = self.motion_model.getMeaStateH()
        self.R = self.motion_model.getMeaNoiseR()

    def inv_unscented_transform(self, sigma_new, weight_c, weight_m, residual_fn=None):
        '''https://git.rst.e-technik.tu-dortmund.de/schuette/common_pytorch_building_blocks/-/blob/master/src/common_pytorch_bb/filter/unscented_transform.py'''
        state_new = sigma_new @ weight_m[..., None]
        if residual_fn is not None:
            sigma_deltas = residual_fn(sigma_new, state_new)
        else:
            sigma_deltas = sigma_new - state_new
        P_new = sigma_deltas @ (torch.diagflat(weight_c) @ sigma_deltas.mT)
        # P_new = torch.sum(outer_products * weight_c[..., None, None], dim=-3)
        return state_new, P_new

    def unscented_transform(self, state, P, alpha, kappa, beta):
        '''https://git.rst.e-technik.tu-dortmund.de/schuette/common_pytorch_building_blocks/-/blob/master/src/common_pytorch_bb/filter/unscented_transform.py'''
        l = state.shape[-2]
        lmbd = alpha**2 * (l + kappa) - l
        matrix_sqrt, info = torch.linalg.cholesky_ex(
            (l + lmbd) * P,
        )  # TODO: cholesky is super unstable in training. replace!!!
        matrix_sqrt = matrix_sqrt.to(state.dtype).to(state.device)
        sigma_i = state + matrix_sqrt.mT
        sigma_iL = state - matrix_sqrt.mT
        sigma = torch.cat((state, sigma_iL, sigma_i), dim=-1)
        weight_m_0 = lmbd / (l + lmbd)
        weight_c_0 = lmbd / (l + lmbd) + (1 - alpha**2 + beta)
        weight_m_i = 1 / (2 * (l + lmbd))
        weight_c_i = 1 / (2 * (l + lmbd))
        weight_m = torch.tensor(
            np.asarray((weight_m_0,) + (weight_m_i,) * 2 * l),
            dtype=state.dtype,
            device=state.device,
        )
        weight_c = torch.tensor(
            np.asarray((weight_c_0,) + (weight_c_i,) * 2 * l),
            dtype=state.dtype,
            device=state.device,
        )
        return sigma, weight_c, weight_m

    def predict(self):
        # predict the state and covariance
        self.state = self.motion_model.getMambaPrediction(self.state)
        self.P = self.F * self.P * self.F.T + self.Q
        self.motion_model.warpStateYawToPi(self.state)

    def predictEgoCentric(self, state):
        self.state = self.motion_model.getMambaPredictionEgoCentric(state)
        self.P = self.F * self.P * self.F.T + self.Q
        self.motion_model.warpStateYawToPi(self.state)

    # unscented transform
    def predictEgoCentricUnscented(self, state):
        l, d = state.shape # l: seq len, d: state dim
        state = torch.from_numpy(state).reshape(l, d, 1) # l d 1
        P = torch.from_numpy(self.P)
        sigma, weightc, weightm = self.unscented_transform(state, P, alpha=0.5, beta=2, kappa=0) # sigma -> l d s // s: no of sigma points (2d+1)

        # use sigma points as batch to parallelize model prediciton
        sigma = rearrange(sigma, 'l d s -> s l d') # s l d
        collect_state = self.motion_model.getMambaPredictionEgoCentricUnscented(sigma)
        collect_state = rearrange(collect_state, 's l d -> l d s').cpu() # l d s

        state, P = self.inv_unscented_transform(collect_state, weightc, weightm) # state -> l d 1 // P -> l d d
        
        # last seq l is the true next predicted step
        state = state[-1].cpu().numpy().reshape(-1)
        self.state = np.mat(state).T
        self.motion_model.warpStateYawToPi(self.state)

        P = P[-1].cpu().numpy()
        self.P = np.mat(P)

    def update(self, curr_det):
        # return if no current det
        if curr_det is None:
            return
        
        # update the state with current det
        measurement = np.mat(curr_det).T
        res = measurement - (self.H * self.state)
        self.motion_model.warpResYawToPi(res)

        S = (self.H * self.P * self.H.T) + self.R
        GAIN = self.P * self.H.T * S.I

        self.state += GAIN * res
        self.P = (np.mat(np.identity(self.n)) - GAIN * self.H) * self.P
        self.motion_model.warpStateYawToPi(self.state)

    def getState(self):
        # returns [x, y, z, w, l, h, vx, vy, ry]
        return self.motion_model.getOutputInfo(self.state)