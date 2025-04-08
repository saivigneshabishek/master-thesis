import numpy as np
from tracking.kalman_filter import KalmanFilter, ExtendedKalmanFilter, MambaFilter
from tracking.life_management import LifeManagement
from tracking.util.transformation import global_to_ego, ego_to_global

class Tracks:
    def __init__(self, cfg, id, init_det, frame_id, ego_data, if_curr_ego):
        self.cfg = cfg
        det_box = init_det[:9]
        self.id = id
        self.cls_id = int(init_det[-3])
        self.score = init_det[-2]
        self.frame_id = frame_id
        self.decay = self.cfg['score_decay'][self.cls_id]
        self.motion_prediction_type = cfg['motion_prediction_type'][self.cls_id]
        self.life = LifeManagement(cfg['life'], self.cls_id, frame_id)
        self.if_curr_ego = if_curr_ego
        self.seq_len = cfg['seq_len']
        self.unscented = cfg['unscented']
        self.updated = False

        # update global history with initial detection
        self.global_history = {}
        global_box = ego_to_global(det_box, ego_data)
        self.global_history[frame_id] = global_box

        if self.motion_prediction_type == 'KalmanFilter':
            self.motion = KalmanFilter(cfg, det_box, self.cls_id)
        elif self.motion_prediction_type == 'ExtendedKalmanFilter':
            self.motion = ExtendedKalmanFilter(cfg, det_box, self.cls_id)
        elif self.motion_prediction_type == 'MambaFilter':
            self.motion = MambaFilter(cfg, det_box, self.cls_id)
        else:
            raise NotImplementedError

    def predict(self, frame_id, ego_data):
        self.score *= self.decay
        self.frame = frame_id
        self.life.predict(frame_id)

        if isinstance(self.motion, MambaFilter):
            if self.if_curr_ego:
                '''
                when tracked in current ego frame, n prev state are transformed to curr ego frame
                and used to predict the present state
                '''
                seq_len = self.seq_len
                if len(self.global_history) < self.seq_len:
                    seq_len = len(self.global_history)
                prev_states = []
                for i in range(frame_id-seq_len, frame_id):
                    prev_state = np.array(global_to_ego(self.global_history[i], ego_data), dtype=np.float32)
                    prev_states.append(prev_state)
                prev_states = np.stack(prev_states, axis=0)
                if self.unscented:
                    self.motion.predictEgoCentricUnscented(prev_states)
                else:
                    self.motion.predictEgoCentric(prev_states)
            else:
                self.motion.predict()
        else:
            if self.if_curr_ego:
                raise Exception("Current EgoTracking not implemented with Kalman, ExtendedKalman Filters")
            self.motion.predict()
        self.updated = False

        # update global history with predicted track
        track_box = self.motion.getState()
        global_box = ego_to_global(track_box, ego_data)
        self.global_history[frame_id] = global_box
 
    def update(self, frame_id, curr_det=None, ego_data=None):
        if curr_det is not None:
            self.score = curr_det[-2]
            curr_det = curr_det[:9]
            self.updated = True
        else:
            self.updated = False
        self.motion.update(curr_det)
        self.life.update(curr_det)

        # update global history with updated track
        track_box = self.motion.getState()
        global_box = ego_to_global(track_box, ego_data)
        self.global_history[frame_id] = global_box

    def getState(self, frame_id=None):
        '''returns state in global coord'''
        return self.global_history[frame_id]
    
    def getLocalState(self, frame_id=None):
        return self.motion.getState()