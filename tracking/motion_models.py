# all the motion models are taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT) and modified to work here

import numpy as np
from tracking.util.utils import warp_to_pi
from model import create_model
import torch

class CV:
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Constant Velocity Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ry]
        Measure vector: [x, y, z, w, l, h, vx, vy, ry]
    """
    def __init__(self, dt: float, qr:dict=None):
        self.dt = dt
        self.n = 10
        self.m = 9
        if qr is None:
            self.q_diag = np.ones(self.n)*100
            self.r_diag = np.ones(self.m)*0.001
        else:
            self.q_diag = [qr['Q_xyz'], qr['Q_xyz'], qr['Q_xyz'], qr['Q_wlh'], qr['Q_wlh'], qr['Q_wlh'], qr['Q_vxy'], qr['Q_vxy'], qr['Q_vxy'], qr['Q_r']]
            self.r_diag = [qr['R_xyz'], qr['R_xyz'], qr['R_xyz'], qr['R_wlh'], qr['R_wlh'], qr['R_wlh'], qr['R_vxy'], qr['R_vxy'], qr['R_r']]

    def getInitState(self, det):
        """from detection init tracklet
        det: [x, y, z, w, l, h, vx, vy, ry]
        """
        state = np.zeros(self.n)
        if det is not None:
            # set x, y, z, w, l, h, vx, vy
            state[:8] = det[:8]
            # set yaw
            state[-1] = det[-1]
        
        state = np.mat(state).T

        return state

    def getInitCovP(self):
        """init errorcov. Generally, the CV model can converge quickly,
        so not particularly sensitive to initialization
        """
        return np.mat(np.eye(self.n)) * 0.01

    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.diag(self.q_diag))

    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.diag(self.r_diag))

    def getTransitionF(self):
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        F = np.mat([[1, 0, 0, 0, 0, 0, self.dt, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, self.dt, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return F

    def getMeaStateH(self):
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return H

    def getOutputInfo(self, state):
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        removes vy and returns [x, y, z, w, l, h, vx, vy, ry]
        """
        state = state.T.tolist()[0]
        # remove vy
        state.pop(-2)
        return np.array(state)

    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state

class CA(): 
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Constant Acceleration Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, vz, ax, ay, az, ry]
        Measure vector: [x, y, z, w, l, h, vx, vy, ry]
    """
    def __init__(self, dt: float):
  
        self.dt = dt
        self.n = 13
        self.m = 9 
    
    def getInitState(self, det):
        """from detection init tracklet
        Acceleration and velocity on the z-axis are both set to 0
        det: [x, y, z, w, l, h, vx, vy, ry]
        """
        state = np.zeros(shape=self.n)
        
        if det is not None:
            # set x, y, z, w, l, h, vx, vy
            state[:8] = det[:8]
            # set yaw
            state[-1] = det[-1]
        
        state = np.mat(state).T
        
        return state
    
    def getInitCovP(self):
        """init errorcov. Generally, the CA model can converge quickly, 
        so not particularly sensitive to initialization
        """
        return np.mat(np.eye(self.n)) * 0.01
    
    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.eye(self.n)) * 100
    
    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.m)) * 0.001
    
    def getTransitionF(self):
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        """
        dt = self.dt
        F = np.mat([[1, 0, 0, 0, 0, 0, dt,  0, 0, 0.5*dt**2,         0,  0, 0],
                    [0, 1, 0, 0, 0, 0,  0, dt, 0,         0, 0.5*dt**2,  0, 0],
                    [0, 0, 1, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 1, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 1, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 1,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  1,  0, 0,        dt,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  1, 0,         0,        dt,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         1,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         1,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 0],
                    [0, 0, 0, 0, 0, 0,  0,  0, 0,         0,         0,  0, 1]])
        return F
    
    def getMeaStateH(self):
        """obtain matrix in the motion_module/script/Linear_kinect_jacobian.ipynb
        """
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        return H

    def getOutputInfo(self, state):
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        removes vy, ax, ay, az and returns [x, y, z, w, l, h, vx, vy, ry]
        """
        state = state.T.tolist()[0]
        # remove [vy, ax, ay, az]
        del state[8:12]
        return np.array(state)
    
    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state

class CTRV:
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Basic info:
        State vector: [x, y, z, w, l, h, v, ry, ry_rate]
        Measure vector: [x, y, z, w, l, h, vx, vy, ry]
    """
    def __init__(self, dt: float):
        self.dt = dt
        self.n = 9
        self.m = 9

    def getInitState(self, det):
        """from detection init tracklet
        Yaw(turn) rate are set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0
        det: [x, y, z, w, l, h, vx, vy, ry]
        """
        state = np.zeros(shape=self.n)

        if det is not None:
            # set x, y, z, w, l, h, v
            state[:6] = det[:6]
            state[6] = np.hypot(det[6], det[7])
            # set yaw
            state[-2] = det[-1]

        state = np.mat(state).T
        return state

    def getInitCovP(self, cov, cls_label=0):
        """init errorcov. In general, when the speed is observable,
        the CTRV model can converge quickly, but when the speed is not measurable,
        we need to carefully set the initial covariance to help the model converge
        """
        # use the same cov as 'car' for every class.
        vector_p = cov[2]

        return np.mat(np.diag(vector_p))

    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.eye(self.n)) * 1

    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.m)) * 1

    def stateTransition(self, state):
        """state transition,
        obtain analytical solutions in the motion_module/script/CTRV_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (9, 1), "state vector number in CTRV must equal to 9"

        dt = self.dt
        x, y, z, w, l, h, v, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v, next_ry = v, theta + omega * dt

        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt
            predict_state = [x + displacement * yaw_cos,
                             y + displacement * yaw_sin,
                             z, w, l, h,
                             next_v,
                             next_ry, omega]
        else:
            ry_rate_inv = 1.0 / omega
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv *
                        ( v * next_yaw_sin - v * yaw_sin),
                             y + ry_rate_inv * ( -v * next_yaw_cos + v * yaw_cos),
                             z, w, l, h,
                             next_v,
                             next_ry, omega]

        return np.mat(predict_state).T

    def StateToMeasure(self, state):
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, vx, vy, ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (9, 1), "state vector number in CTRV must equal to 9"

        x, y, z, w, l, h, v, theta, _ = state.T.tolist()[0]
        state_info = [x, y, z,
                        w, l, h,
                        v * np.cos(theta),
                        v * np.sin(theta),
                        theta]

        return np.mat(state_info).T

    def getTransitionF(self, state):
        """obtain matrix in the motion_module/script/CTRV_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.dt
        _, _, _, _, _, _, v, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)

        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            F = np.mat([[1, 0, 0, 0, 0, 0, dt * yaw_cos, -dt * v * yaw_sin, 0],
                        [0, 1, 0, 0, 0, 0, dt * yaw_sin,  dt * v * yaw_cos, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, dt],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        else:
            ry_rate_inv, ry_rate_inv_square = 1 / omega, 1 / (omega * omega)
            next_ry = theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            F = np.mat([[1, 0, 0, 0, 0, 0, -ry_rate_inv * (yaw_sin - next_yaw_sin),
                          ry_rate_inv * (v * next_yaw_cos - v * yaw_cos),
                            ry_rate_inv_square * (v * yaw_sin - v * next_yaw_sin) + ry_rate_inv * dt * v * next_yaw_cos],
                        [0, 1, 0, 0, 0, 0, ry_rate_inv * (yaw_cos - next_yaw_cos),
                         ry_rate_inv * (v * next_yaw_sin - v * yaw_sin),
                            ry_rate_inv_square * (v * next_yaw_cos - v * yaw_cos) + ry_rate_inv * dt * v * next_yaw_sin],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, dt],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        return F

    def getMeaStateH(self, state):
        """obtain matrix in the motion_module/script/CTRV_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """
        _, _, _, _, _, _, v, theta, _ = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, yaw_cos, -v * yaw_sin, 0],
                    [0, 0, 0, 0, 0, 0, yaw_sin, v * yaw_cos, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0]])

        return H

    def getOutputInfo(self, state):
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        returns [x, y, z, w, l, h, vx, vy, ry]
        """
        x, y, z, w, l, h, v, theta, _ = state.T.tolist()[0]
        ret_state = [x, y, z, w, l, h, v*np.cos(theta), v*np.sin(theta), theta]

        return np.array(ret_state)
    
    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state

class CTRA: 
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, v, a, ry, ry_rate]
        Measure vector: [x, y, z, w, l, h, vx, vy, ry]
    """
    def __init__(self, dt: float):
        self.dt = dt
        self.n = 10
        self.MD = 9
        
    def getInitState(self, det):
        """from detection init tracklet
        Acceleration and yaw(turn) rate are both set to 0. when velociy
        on X/Y-Axis are available, the combined velocity is also set to 0
        det: [x, y, z, w, l, h, vx, vy, ry]
        """
        state = np.zeros(shape=self.n)

        if det is not None:
            # set x, y, z, w, l, h, v
            state[:6] = det[:6]
            state[6] = np.hypot(det[6], det[7])
            # set yaw
            state[-2] = det[-1]

        state = np.mat(state).T
        return state
    
    def getInitCovP(self, cov, cls_label=0):
        """init errorcov. In general, when the speed is observable, 
        the CTRA model can converge quickly, but when the speed is not measurable, 
        we need to carefully set the initial covariance to help the model converge
        """
        # use the same 'cov as 'car' for every class.
        vector_p = cov[2]
        
        return np.mat(np.diag(vector_p))
    
    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.eye(self.n)) * 1
    
    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.MD)) * 1
    
    def stateTransition(self, state):
        """state transition, 
        obtain analytical solutions in the motion_module/script/CTRA_kinect_jacobian.ipynb
        Args:
            state (np.mat): [state dim, 1] the estimated state of the previous frame

        Returns:
            np.mat: [state dim, 1] the predict state of the current frame
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        dt = self.dt
        x, y, z, w, l, h, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        next_v, next_ry = v + a * dt, theta + omega * dt
        
        # corner case(tiny yaw rate), prevent divide-by-zero overflow
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x + displacement * yaw_cos,
                             y + displacement * yaw_sin,
                             z, w, l, h, 
                             next_v, a, 
                             next_ry, omega]
        else:
            ry_rate_inv_square = 1.0 / (omega * omega)
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            predict_state = [x + ry_rate_inv_square * (next_v * omega * next_yaw_sin + a * next_yaw_cos - v * omega * yaw_sin - a * yaw_cos),
                             y + ry_rate_inv_square * (-next_v * omega * next_yaw_cos + a * next_yaw_sin + v * omega * yaw_cos - a * yaw_sin),
                             z, w, l, h,
                             next_v, a, 
                             next_ry, omega]
        
        return np.mat(predict_state).T  
    
    def StateToMeasure(self, state):
        """get state vector in the measure space
        state vector -> [x, y, z, w, l, h, v, a, ry, ry_rate]
        measure space -> [x, y, z, w, l, h, vx, vy, ry]

        Args:
            state (np.mat): [state dim, 1] the predict state of the current frame

        Returns:
            np.mat: [measure dim, 1] state vector projected in the measure space
        """
        assert state.shape == (10, 1), "state vector number in CTRA must equal to 10"
        
        x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        state_info = [x, y, z,
                    w, l, h,
                    v * np.cos(theta),
                    v * np.sin(theta),
                    theta]
        
        return np.mat(state_info).T
    
    def getTransitionF(self, state):
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        dt = self.dt
        _, _, _, _, _, _, v, a, theta, omega = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        
        # corner case, tiny turn rate
        if abs(omega) < 0.001:
            displacement = v * dt + a * dt ** 2 / 2
            F = np.mat([[1, 0, 0, 0, 0, 0,  dt*yaw_cos,   dt**2*yaw_cos/2,        -displacement*yaw_sin,  0],
                        [0, 1, 0, 0, 0, 0,  dt*yaw_sin,   dt**2*yaw_sin/2,         displacement*yaw_cos,  0],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 1,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           1,                dt,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 1,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            1, dt],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            0,  1]])
        else:
            ry_rate_inv, ry_rate_inv_square, ry_rate_inv_cube = 1 / omega, 1 / (omega * omega), 1 / (omega * omega * omega)
            next_v, next_ry = v + a * dt, theta + omega * dt
            next_yaw_sin, next_yaw_cos = np.sin(next_ry), np.cos(next_ry)
            F = np.mat([[1, 0, 0, 0, 0, 0,  -ry_rate_inv*(yaw_sin-next_yaw_sin),   -ry_rate_inv_square*(yaw_cos-next_yaw_cos)+ry_rate_inv*dt*next_yaw_sin,        ry_rate_inv_square*a*(yaw_sin-next_yaw_sin)+ry_rate_inv*(next_v*next_yaw_cos-v*yaw_cos),  ry_rate_inv_cube*2*a*(yaw_cos-next_yaw_cos)+ry_rate_inv_square*(v*yaw_sin-v*next_yaw_sin-2*a*dt*next_yaw_sin)+ry_rate_inv*dt*next_v*next_yaw_cos ],
                        [0, 1, 0, 0, 0, 0,   ry_rate_inv*(yaw_cos-next_yaw_cos),   -ry_rate_inv_square*(yaw_sin-next_yaw_sin)-ry_rate_inv*dt*next_yaw_cos,        ry_rate_inv_square*a*(-yaw_cos+next_yaw_cos)+ry_rate_inv*(next_v*next_yaw_sin-v*yaw_sin), ry_rate_inv_cube*2*a*(yaw_sin-next_yaw_sin)+ry_rate_inv_square*(v*next_yaw_cos-v*yaw_cos+2*a*dt*next_yaw_cos)+ry_rate_inv*dt*next_v*next_yaw_sin ],
                        [0, 0, 1, 0, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 1, 0, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 1, 0,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 1,           0,                 0,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           1,                dt,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 1,                            0,  0],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            1, dt],
                        [0, 0, 0, 0, 0, 0,           0,                 0,                            0,  1]])
                                
        return F
    
    def getMeaStateH(self, state):
        """obtain matrix in the motion_module/script/CTRA_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """
        _, _, _, _, _, _, v, _, theta, _ = state.T.tolist()[0]
        yaw_sin, yaw_cos = np.sin(theta), np.cos(theta)
        H = np.mat([[1, 0, 0, 0, 0, 0,        0, 0,           0, 0],
                    [0, 1, 0, 0, 0, 0,        0, 0,           0, 0],
                    [0, 0, 1, 0, 0, 0,        0, 0,           0, 0],
                    [0, 0, 0, 1, 0, 0,        0, 0,           0, 0],
                    [0, 0, 0, 0, 1, 0,        0, 0,           0, 0],
                    [0, 0, 0, 0, 0, 1,        0, 0,           0, 0],
                    [0, 0, 0, 0, 0, 0, yaw_cos,  0,  -v*yaw_sin, 0],
                    [0, 0, 0, 0, 0, 0, yaw_sin,  0,   v*yaw_cos, 0],
                    [0, 0, 0, 0, 0, 0,        0, 0,           1, 0]])

        return H

    def getOutputInfo(self, state) -> np.array:
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        returns [x, y, z, w, l, h, vx, vy, ry]
        """
        x, y, z, w, l, h, v, _, theta, _ = state.T.tolist()[0]
        ret_state = [x, y, z, w, l, h, v * np.cos(theta), v * np.sin(theta), theta]
        
        return np.array(ret_state)
    
    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state

class Bicycle:
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Constant Acceleration and Turn Rate Motion Model
    Basic info:
        State vector: [x_gra, y_gra, z_geo, w, l, h, v, a, ry, sigma]
        Measure vector: [x_geo, y_geo, z_geo, w, l, h, vx, vy, ry]
    Important assumptions:
        1. Although the acceleration interface is reserved, 
        we still think that the velocity is constant when 
        beta is large. In other cases, the object is considered 
        to be moving in a straight line with uniform acceleration.
        2. Based on experience, we set two hyperparameters here, 
        wheelbase ratio is set to 0.8, rear tire ratio is set to 0.5.
        3. the steering angle(sigma) is also considered to be constant.
        4. x, y described in the filter are gravity center, however, 
        the infos we measure are geometric center. Transformation 
        process is required.
        5. We don't intergrate variable 'length' to the jacobian matrix
        for simple and fast solution
    """
    def __init__(self, dt: float):
        self.dt = dt
        self.n = 10
        self.m = 9
        self.w_r, self.lf_r = 0.8, 0.5
        
    def getInitState(self, det):
        """from detection init tracklet, we set some assumptions in 
        the BICYCLE model for easy calculation
        det: [x_geo, y_geo, z_geo, w, l, h, vx, vy, ry]
        """
        state = np.zeros(shape=self.n)
        
        if det is not None:
            # set gravity center x, y
            state[:2] = self.geoCenterToGraCenter(geo_center=[det[0], det[1]],
                                                    theta=det[-1],
                                                    length=det[4])
            
            # set z, w, l, h, (v, if velo is valid)
            state[2:6] = det[2:6]
            state[6] = np.hypot(det[6], det[7])
            # set yaw
            state[-2] = det[-1]

        state = np.mat(state).T
        return state
    
    def getInitCovP(self, cov, cls_label=0):
        """init errorcov. In the BICYCLE motion model, we apply 
        same errorcov initialization strategy as the CTRA model.
        """

        vector_p = cov[cls_label] if cls_label == 0 else [10, 10, 10, 10, 10, 10, 1000, 10, 10, 10]
        
        return np.mat(np.diag(vector_p))
    
    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.eye(self.n)) * 1
    
    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.eye(self.m)) * 1
    
    def getTransitionF(self, state):
        """obtain matrix in the motion_module/script/BIC_kinect_jacobian.ipynb
        d(stateTransition) / d(state) at previous_state
        """
        
        dt = self.dt
        _, _, _, _, l, _, v, a, theta, sigma = state.T.tolist()[0]
        beta, _, lr = self.getBicBeta(l, sigma)
        
        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)
        
        # corner case, tiny beta
        if abs(beta) < 0.001:
            displacement = a*dt**2/2 + dt*v
            F = np.mat([[1, 0, 0, 0, 0, 0,  dt*cos_yaw,  dt**2*cos_yaw/2,        -displacement*sin_yaw, 0],
                        [0, 1, 0, 0, 0, 0,  dt*sin_yaw,  dt**2*sin_yaw/2,         displacement*cos_yaw, 0],
                        [0, 0, 1, 0, 0, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 1, 0, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 1, 0,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 0, 1,           0,                0,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           1,               dt,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           0,                1,                            0, 0],
                        [0, 0, 0, 0, 0, 0,           0,                0,                            1, 0],
                        [0, 0, 0, 0, 0, 0,           0,                0,                            0, 1]])
        else:
            next_yaw, sin_beta = theta + v / lr * np.sin(beta) * dt, np.sin(beta)
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            F = np.mat([[1, 0, 0, 0, 0, 0, dt*np.cos(next_v_yaw), 0, -lr*np.cos(v_yaw)/sin_beta + lr*np.cos(next_v_yaw)/sin_beta, 0],
                        [0, 1, 0, 0, 0, 0, dt*np.sin(next_v_yaw), 0, -lr*np.sin(v_yaw)/sin_beta + lr*np.sin(next_v_yaw)/sin_beta, 0],
                        [0, 0, 1, 0, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 1, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 1, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 1,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,                     1, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,                     0, 0,                                                           0, 0],
                        [0, 0, 0, 0, 0, 0,        dt/lr*sin_beta, 0,                                                           1, 0],
                        [0, 0, 0, 0, 0, 0,                     0, 0,                                                           0, 1]])
        return F
    
    def getMeaStateH(self, state):
        """obtain matrix in the motion_module/script/BIC_kinect_jacobian.ipynb
        d(StateToMeasure) / d(state) at predict_state
        """
        _, _, _, _, l, _, v, _, theta, sigma = state.T.tolist()[0]
        
        geo2gra_dist, lr2l = self.graToGeoDist(l), self.w_r * (0.5 - self.lf_r)
        sin_yaw, cos_yaw = np.sin(theta), np.cos(theta)
        
        beta, _, _ = self.getBicBeta(l, sigma)
        v_yaw = beta + theta
        sin_v_yaw, cos_v_yaw = np.sin(v_yaw), np.cos(v_yaw)
        H = np.mat([[1, 0, 0, 0, -lr2l*cos_yaw, 0,               0, 0,  geo2gra_dist*sin_yaw, 0],
                    [0, 1, 0, 0, -lr2l*sin_yaw, 0,               0, 0, -geo2gra_dist*cos_yaw, 0],
                    [0, 0, 1, 0,             0, 0,               0, 0,                     0, 0],
                    [0, 0, 0, 1,             0, 0,               0, 0,                     0, 0],
                    [0, 0, 0, 0,             1, 0,               0, 0,                     0, 0],
                    [0, 0, 0, 0,             0, 1,               0, 0,                     0, 0],
                    [0, 0, 0, 0,             0, 0,       cos_v_yaw, 0,          -v*sin_v_yaw, 0],
                    [0, 0, 0, 0,             0, 0,       sin_v_yaw, 0,           v*cos_v_yaw, 0],
                    [0, 0, 0, 0,             0, 0,               0, 0,                     1, 0]])
        
        return H
    
    def stateTransition(self, state):
        """State transition based on model assumptions
        """
        assert state.shape == (10, 1), "state vector number in BICYCLE must equal to 10"
        
        dt = self.dt
        x_gra, y_gra, z, w, l, h, v, a, theta, sigma = state.T.tolist()[0]
        beta, _, lr = self.getBicBeta(l, sigma)
        
        # corner case, tiny yaw rate
        if abs(beta) > 0.001:
            next_yaw = theta + v / lr * np.sin(beta) * dt
            v_yaw, next_v_yaw = beta + theta, beta + next_yaw
            predict_state = [x_gra + (lr * (np.sin(next_v_yaw) - np.sin(v_yaw))) / np.sin(beta),  # x_gra
                             y_gra - (lr * (np.cos(next_v_yaw) - np.cos(v_yaw))) / np.sin(beta),  # y_gra
                             z, w, l, h,
                             v, 0, next_yaw, sigma]
        else:
            displacement = v * dt + a * dt ** 2 / 2
            predict_state = [x_gra + displacement * np.cos(theta),
                             y_gra + displacement * np.sin(theta),
                             z, w, l, h,
                             v + a * dt, a, theta, sigma]
        return np.mat(predict_state).T  
    
    def StateToMeasure(self, state):
        """get state vector in the measure space
        """
        assert state.shape == (10, 1), "state vector number in BICYCLE must equal to 10"
        
        x_gra, y_gra, z, w, l, h, v, _, theta, sigma = state.T.tolist()[0]
        
        beta, _, _ = self.getBicBeta(l, sigma)
        geo2gra_dist = self.graToGeoDist(l)
        
        meas_state = [x_gra - geo2gra_dist * np.cos(theta),
                        y_gra - geo2gra_dist * np.sin(theta),
                        z, w, l, h,
                        v * np.cos(theta + beta),
                        v * np.sin(theta + beta),
                        theta]
        
        return np.mat(meas_state).T
    
    def getBicBeta(self, length: float, sigma: float):
        """get the angle between the object velocity and the 
        X-axis of the coordinate system

        Args:
            length (float): object length
            sigma (float): the steering angle, radians

        Returns:
            float: the angle between the object velocity and X-axis, radians
        """
        
        lf, lr = length * self.w_r * self.lf_r, length * self.w_r * (1 - self.lf_r)
        beta = np.arctan(lr / (lr + lf) * np.tan(sigma))
        return beta, lf, lr
        
    
    def geoCenterToGraCenter(self, geo_center: list, theta: float, length: float):
        """from geo center to gra center

        Args:
            geo_center (list): object geometric center
            theta (float): object heading yaw
            length (float): object length

        Returns:
            np.ndarray: object gravity center
        """
        geo2gra_dist = self.graToGeoDist(length)
        gra_center = [geo_center[0] + geo2gra_dist * np.cos(theta),
                      geo_center[1] + geo2gra_dist * np.sin(theta)]
        return np.array(gra_center)
    
    def graCenterToGeoCenter(self, gra_center: list, theta: float, length: float):
        """from gra center to geo center

        Args:
            gra_center (list): object gravity center
            theta (float): object heading yaw
            length (float): object length

        Returns:
            np.ndarray: object geometric center
        """
        geo2gra_dist = self.graToGeoDist(length)
        geo_center = [gra_center[0] - geo2gra_dist * np.cos(theta),
                      gra_center[1] - geo2gra_dist * np.sin(theta)]
        return np.array(geo_center)
    
    def graToGeoDist(self, length: float):
        """get gra center to geo center distance

        Args:
            length (float): object length

        Returns:
            float: gra center to geo center distance
        """
        return length * self.w_r * (0.5 - self.lf_r)

    def getOutputInfo(self, state):
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        returns [x, y, z, w, l, h, vx, vy, ry]
        """
        x_gra, y_gra, z, w, l, h, v, _, theta, sigma = state.T.tolist()[0]
        beta, _, _ = self.getBicBeta(l, sigma)
        x, y = self.graCenterToGeoCenter(gra_center=[state[0, 0], state[1, 0]],
                                               theta=state[-2, 0],
                                               length=state[4, 0])
        
        ret_state = [x, y, z, w, l, h, v*np.cos(theta+beta), v*np.sin(theta+beta), theta]
        
        return np.array(ret_state)
    
    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-2, 0] = warp_to_pi(state[-2, 0])
        return state
    
class SSM:
    """
    model taken from Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)
    Constant Velocity Motion Model
    Basic info:
        State vector: [x, y, z, w, l, h, vx, vy, ry]
        Measure vector: [x, y, z, w, l, h, vx, vy, ry]
    """
    def __init__(self, dt: float, cfg:dict):
        self.dt = dt
        self.n = 9
        self.m = 9

        # self.q_diag = [0.7419012989381537, 0.7419012989381537, 0.7419012989381537, 0.8646864641426516, 0.8646864641426516, 0.8646864641426516, 0.19561616226862616, 0.19561616226862616, 0.9451318561249286]
        # self.r_diag = [0.3388479618872789, 0.3388479618872789, 0.3388479618872789, 0.592473716146918, 0.592473716146918, 0.592473716146918, 0.0760055189000628, 0.0760055189000628, 0.07708204452298678]
        
        ## MAMBA_RESIDUAL
        self.q_diag = [0.9584907787344648, 0.9584907787344648, 0.9584907787344648, 0.1496854527819701, 0.1496854527819701, 0.1496854527819701, 0.324104189671782, 0.324104189671782, 0.7775016681062613]
        self.r_diag = [0.9732804959990586, 0.9732804959990586, 0.9732804959990586, 0.6199891719658087, 0.6199891719658087, 0.6199891719658087, 0.12901846013937154, 0.12901846013937154, 0.025604112233310185]

        ## MAMBA_BASE
        # self.q_diag = [0.7533569252163739, 0.7533569252163739, 0.7533569252163739, 0.1275911020475583, 0.1275911020475583, 0.1275911020475583, 0.5200107536709833, 0.5200107536709833, 0.377710782673907]
        # self.r_diag = [0.4020568205414047, 0.4020568205414047, 0.4020568205414047, 0.8092686280547359, 0.8092686280547359, 0.8092686280547359, 0.2771301138684882, 0.2771301138684882, 0.02261138955614751]

        # qr =  cfg.get('MAMBA_Tuning')
        # if qr is None:
        #     self.q_diag = np.ones(self.n)*100
        #     self.r_diag = np.ones(self.m)*0.001
        # else:
        #     self.q_diag = [qr['Q_xyz'], qr['Q_xyz'], qr['Q_xyz'], qr['Q_wlh'], qr['Q_wlh'], qr['Q_wlh'], qr['Q_vxy'], qr['Q_vxy'], qr['Q_r']]
        #     self.r_diag = [qr['R_xyz'], qr['R_xyz'], qr['R_xyz'], qr['R_wlh'], qr['R_wlh'], qr['R_wlh'], qr['R_vxy'], qr['R_vxy'], qr['R_r']]

        # init model and load weights
        self.mamba = create_model(cfg['ssm_model'])
        ckpt = torch.load(cfg['checkpoint'])
        ckpt = ckpt['model'] if 'model' in ckpt.keys() else ckpt
        self.mamba.load_state_dict(ckpt)
        self.mamba.eval()

    def getInitState(self, det):
        """from detection init tracklet
        det: [x, y, z, w, l, h, vx, vy, ry]
        """
        state = np.zeros(self.n)
        if det is not None:
            # set x, y, z, w, l, h, vx, vy, ry
            state = det
        
        state = np.mat(state).T

        return state

    def getInitCovP(self):
        """init errorcov. Generally, the CV model can converge quickly,
        so not particularly sensitive to initialization
        """
        return np.mat(np.eye(self.n)) * 0.01

    def getProcessNoiseQ(self):
        """set process noise(fix)
        """
        return np.mat(np.diag(self.q_diag))

    def getMeaNoiseR(self):
        """set measure noise(fix)
        """
        return np.mat(np.diag(self.r_diag))
    
    def getMambaPrediction(self, state):
        return self.mamba.predict(state)
    
    def getMambaPredictionEgoCentric(self, state):
        return self.mamba.predictEgoCentric(state)

    def getMambaPredictionEgoCentricUnscented(self, state):
        return self.mamba.predictEgoCentricUnscented(state)
    
    def getTransitionF(self):
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        F = np.mat([[1, 0, 0, 0, 0, 0, self.dt, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, self.dt, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return F

    def getMeaStateH(self):
        """obtain matrix in the motion_module/script/CV_kinect_jacobian.ipynb
        """
        H = np.mat([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return H

    def getOutputInfo(self, state):
        """convert state vector in the filter to the output format
        Note that, tra score will be process later
        removes vy and returns [x, y, z, w, l, h, vx, vy, ry]
        """
        state = state.T.tolist()[0]
        return np.array(state)

    @staticmethod
    def warpResYawToPi(res):
        """warp res yaw to [-pi, pi) in place

        Args:
            res (np.mat): [measure dim, 1]
            res infos -> [x, y, z, w, l, h, (vx, vy, optional), ry]

        Returns:
            np.mat: [measure dim, 1], residual warped to [-pi, pi)
        """
        res[-1, 0] = warp_to_pi(res[-1, 0])
        return res

    @staticmethod
    def warpStateYawToPi(state):
        """warp state yaw to [-pi, pi) in place

        Args:
            state (np.mat): [state dim, 1]
            State vector: [x, y, z, w, l, h, vx, vy, vz, ry]

        Returns:
            np.mat: [state dim, 1], state after warping
        """
        state[-1, 0] = warp_to_pi(state[-1, 0])
        return state