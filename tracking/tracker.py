import numpy as np
from tracking.tracks import Tracks
from tracking.data_association import matching
from tracking.util.distance import giou_3d
from tracking.util.nms import blend_nms
from tracking.util.utils import reorder_metrics, mask_tras_dets, stack_tracks, warp_to_pi
from tracking.util.transformation import global_to_ego
from pyquaternion import Quaternion
from typing import Tuple, List, Union
from nuscenes.nuscenes import NuScenes

class MultiObjectTracker:
    '''
    taken from Poly-MOT and modified (https://github.com/lixiaoyu2000/Poly-MOT)
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.cls_num = cfg['num_class']
        self.two_stage = cfg['association']['two_stage']
        self.metrics = cfg['association']['metrics']
        self.re_metrics = reorder_metrics(self.metrics)
        self.ego_transform = cfg['ego_transform']
        self.ego_frame = cfg['ego_frame']
        if self.ego_transform:
            self.nusc = NuScenes(version=cfg['nusc']['version'], dataroot=cfg['nusc']['path'], verbose=False)
        assert self.ego_frame in ['first', 'current', 'last'], "EgoFrame transform must either of: [first, last, current]."
        self.reset()

    def reset(self):
        self.active_tracks = {}
        self.dead_tracks = {}
        self.tentative_tracks = {}
        self.valid_tracks = {}
        self.all_valid_tracks = []
        self.id_seed = 0
        self.detections = None
        self.ego_data = None

    def tracking(self, frame_data):

        # if first frame, reset all states of the tracker
        if frame_data['frame_id'] == 0:
            self.reset()
        self.frame_data = frame_data
        self.frame_id = int(self.frame_data['frame_id'])

        # corner case: no detections in current frame
        if int(self.frame_data['num_dets']) == 0:
            if len(self.valid_tracks) == 0:
                return None
            write_db = self.update(ids=np.array([]))
            if len(write_db) == 0:
                return None
            return write_db
        
        self.detections = self.frame_data['dets'][0].numpy() # <---- [1,N,12]
        # apply NMS
        self.detections = blend_nms(self.detections, self.cfg['nms'])
        self.det_num = len(self.detections)

        # ego transformations
        if self.ego_transform:
            if self.ego_frame == 'first':
                token = self.frame_data['first_sample'][0]
            elif self.ego_frame == 'last':
                token = self.frame_data['last_sample'][0]
            else:
                token = self.frame_data['sample'][0]
        
            sample = self.nusc.get('sample', token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sd_record = self.nusc.get('sample_data', sample_data_token)
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

            # detections/tracks are rotated only in z axis, set x and y components to 0
            pose_record['rotation'][1] = 0.0
            pose_record['rotation'][2] = 0.0
            # get transformation matrix global->ego / ego->global
            self.ego_data = {
                'ego_translation': pose_record['translation'],
                'to_ego': Quaternion(pose_record['rotation']).inverse,
                'to_global': Quaternion(pose_record['rotation'])}
            
            # transform detections to ego frame
            detections = []
            for box in self.detections:
                det = box[:9]
                box[:9] = global_to_ego(det, self.ego_data)
                detections.append(box)
            self.detections = np.stack(detections, axis=0)

        # predict current tracks from prev tracks
        self.prediction()

        # matched, unmatched_dets, unmatched_tracks = self.association()
        ids = self.association()

        # update the tracks with current info and write the tracking results
        write_db = self.update(ids)

        # only for debug
        assert len(self.tentative_tracks) == 0, "no tentative tracklet in the best performance version."

        return write_db
    
    def prediction(self):
        self.all_valid_tracks = []
        # when first frame, there are no prev tracks
        if len(self.valid_tracks) == 0: return
        # for every available prev track, predict its future position
        for track_idx, track in self.valid_tracks.items():
            track.predict(frame_id=self.frame_id, ego_data=self.ego_data)
            self.all_valid_tracks.append(track_idx)

        assert len(self.all_valid_tracks) == len(self.valid_tracks)

    def update(self, ids):
        tracking_ids = ids.tolist()
        assert len(tracking_ids) == self.det_num

        new_track, tentative_track, active_track = {}, {}, {}
        results = []

        for det_idx, track_idx in enumerate(tracking_ids):
            det = self.detections[det_idx]
            assert track_idx not in self.dead_tracks

            # update matched tracks with current info
            if track_idx in self.valid_tracks:
                track = self.valid_tracks[track_idx]
                track.update(frame_id=self.frame_id, curr_det=det, ego_data=self.ego_data)

            # unmatched detections: initialize new tracks
            else:
                track = Tracks(cfg=self.cfg['track'],
                               id=track_idx,
                               init_det=det,
                               frame_id=self.frame_id,
                               ego_data=self.ego_data,
                               if_curr_ego=True if (self.ego_transform and self.ego_frame == 'current') else False)
                new_track[track_idx] = track

        temp_merge_track = {**self.valid_tracks, **new_track}

        for track_idx, track in temp_merge_track.items():
            # updated unmatched tracks with no current info
            if track_idx not in tracking_ids:
                track.update(frame_id=self.frame_id, curr_det=None, ego_data=self.ego_data)

            # write tracking results to db
            if track.life.state == 'active':
                active_track[track_idx] = track
                '''
                only Tracks that are updated in the current frame and tracks that are not
                updated until last_update_limit frames are written to the results
                '''
                if track.updated is True:
                    ret = self.write_db(track)
                    results.append(ret)
                elif track.life.time_since_last_update <= self.cfg['last_update_limit']:
                    ret = self.write_db(track)
                    results.append(ret)
            
            elif track.life.state == 'tentative':
                tentative_track[track_idx] = track
            elif track.life.state == 'dead':
                assert track_idx not in self.dead_tracks
                self.dead_tracks[track_idx] = track
            else: raise Exception('Tracks can only have three states: active, tentative, dead')

        # update for next iter
        self.active_tracks, self.tentative_tracks = active_track, tentative_track
        self.valid_tracks = {**self.active_tracks, **self.tentative_tracks}

        return results

    def association(self):
        '''returns indices of matched, unmatched detections and tracks'''
        if len(self.valid_tracks) == 0:
            ids = np.arange(self.id_seed, self.id_seed+self.det_num, dtype=int)
            self.id_seed += self.det_num
        else:
            cost_matrices = self.compute_cost()
            ids = self.matching_cost(cost_matrices)
        return ids
    
    def compute_cost(self) -> dict:
        """
        taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
        Construct the cost matrix between the trajectory and the detection
        :return: dict, a collection of cost matrices,
        one-stage: np.array, [cls_num, det_num, tra_num], two-stage: np.array, [det_num, tra_num]
        """
        tracks = stack_tracks(tracks=self.valid_tracks, frame_id=self.frame_id)
        # [N, 1] containing the class id
        det_labels, tra_labels = self.detections[:, -3], tracks[:, -3]

        # [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
        valid_mask = mask_tras_dets(self.cls_num, det_labels, tra_labels)

        two_cost, first_cost = giou_3d(self.detections, tracks)
        first_cost = first_cost[None, :, :].repeat(self.cls_num, axis=0)
        first_cost[self.re_metrics['giou_bev']] = two_cost

        # mask invalid value
        first_cost[np.where(~valid_mask)] = -np.inf

        # Due to the execution speed of python,
        # construct the two-stage cost matrix under half-parallel framework is very tricky, 
        # we strongly recommend only use giou_bev as two-stage metric to build the cost matrix
        return {'one_stage': 1 - first_cost, 'two_stage': 1 - two_cost if two_cost is not None else None}

    def matching_cost(self, cost_matrices: dict) -> np.array:
        """
        taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
        Solve the matching pair according to the cost matrix
        :param cost_matrices: cost matrices between dets and tras construct in the one/two stage
        :return: np.array, tracking id of each detection
        """
        cost1, cost2 = cost_matrices['one_stage'], cost_matrices['two_stage']
        # m_tras_1 is not the tracking id, but is the index of tracklet in the all valid trajectories
        m_dets_1, m_tras_1, um_dets_1, um_tras_1 = matching(self.cfg['association'], cost1, stage_two=False)
        if self.two_stage:
            inf_cost = np.ones_like(cost2) * np.inf
            inf_cost[np.ix_(um_dets_1, um_tras_1)] = 0
            cost2 += inf_cost
            m_dets_2, m_tras_2, _, _ = matching(self.cfg['association'], cost2, stage_two=True)
            m_dets_1 += m_dets_2
            m_tras_1 += m_tras_2

        assert len(m_dets_1) == len(m_tras_1), "as the pair, number of the matched tras and dets must be equal"
        # corner case, no matching pair after matching
        if len(m_dets_1) == 0:
            ids = np.arange(self.id_seed, self.id_seed + self.det_num, dtype=int)
            self.id_seed += self.det_num
            return ids
        else:
            ids, match_pairs = [], {key: value for key, value in zip(m_dets_1, m_tras_1)}
            all_valid_ids = self.all_valid_tracks
            for det_idx in range(self.det_num):
                if det_idx not in m_dets_1:
                    ids.append(self.id_seed)
                    self.id_seed += 1
                else:
                    ids.append(all_valid_ids[match_pairs[det_idx]])

        return np.array(ids)

    def write_db(self, track):
        box = np.array(track.getState(self.frame_id))
        id = track.id
        cls_id = track.cls_id
        score = track.score
        yaw = box[8]

        # to be sure the global quaternion is in correct axis
        rot_matrix = Quaternion(axis=(0, 0, 1), radians=yaw).rotation_matrix
        rot = Quaternion(matrix=rot_matrix)

        ret = {
            "sample_token": self.frame_data['sample'][0],
            "translation": box[:3].tolist(),
            "size": box[3:6].tolist(),
            "velocity": box[6:8].tolist(),
            "rotation": rot.q.tolist(),
            "tracking_id": id,
            "tracking_name": self.cfg['class'][cls_id],
            "tracking_score": float(score),
        }

        return ret