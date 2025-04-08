
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Union
from tracking.util.distance import giou3d

def matching(cfg, cost_fn, stage_two=False):
    '''
    NOT USED AND NOT UPDATED
    Calls the matching algorithm and returns matched and unmatched dets, tracks'''
    threshold = cfg['threshold2'] if stage_two else cfg['threshold1']
    algorithm = cfg['algorithm']
    if algorithm == 'Hungarian':
        return Hungarian(cost_fn, threshold)
    elif algorithm == 'Greedy':
        return Greedy(cost_fn, threshold)
    elif algorithm == 'MNN':
        return MNN(cost_fn, threshold)
    else:
        raise NotImplementedError

def association(cfg, dets, tracks):
    '''
    NOT USED AND NOT UPDATED
    data association method taken from SimpleTrack (https://github.com/tusen-ai/SimpleTrack)'''
    cost_matrix = np.zeros((len(dets), len(tracks)))
    for d, det in enumerate(dets):
        for t, track in enumerate(tracks):
                cost_matrix[d,t] = giou3d(det, track.getState())
    cost_matrix = 1 - cost_matrix

    unmatched_dets = []
    unmatched_tracks = []

    row , col = linear_sum_assignment(cost_matrix)
    matched_indices = np.stack([row, col], axis=1)
    for d, _ in enumerate(dets):
         if d not in matched_indices[:,0]:
              unmatched_dets.append(d)
    
    for t, _ in enumerate(tracks):
         if t not in matched_indices[:,1]:
              unmatched_tracks.append(t)
    
    matched = []
    for idx in matched_indices:
        obj_id = int(dets[idx[0]][-3])
        if cost_matrix[idx[0], idx[1]] > cfg['threshold'][obj_id]:
            unmatched_dets.append(idx[0])
            unmatched_tracks.append(idx[1])
        else:
            matched.append(idx.reshape(2))

    return matched, np.array(unmatched_dets), np.array(unmatched_tracks)        

def Hungarian(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """
    taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
    implement hungarian algorithm with lap

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be equal to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    for cls_idx, cls_cost in enumerate(cost_matrix):
        _, x, y = lap.lapjv(cls_cost, extend_cost=True, cost_limit=thresholds[cls_idx])
        for ix, mx in enumerate(x):
            if mx >= 0:
                assert (ix not in m_det) and (mx not in m_tra) 
                m_det.append(ix)
                m_tra.append(mx)
                            
    # unmatched tra and det
    num_det, num_tra = cost_matrix.shape[1:]
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra

def Greedy(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """
    taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
    implement greedy algorithm

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be egual to cost matrix number."
    
    # solve cost matrix
    m_det, m_tra = [], []
    num_det, num_tra = cost_matrix.shape[1:] 
    for cls_idx, cls_cost in enumerate(cost_matrix):
        for det_idx in range(num_det):
            tra_idx = cls_cost[det_idx].argmin()
            if cls_cost[det_idx][tra_idx] <= thresholds[cls_idx]:
                cost_matrix[cls_idx, :, tra_idx] = 1e18
                m_det.append(det_idx)
                m_tra.append(tra_idx)
    
    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))
    
    return m_det, m_tra, um_det, um_tra

def MNN(cost_matrix: np.array, thresholds: dict) -> Tuple[list, list, np.array, np.array]:
    """
    taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
    implement MNN(Mutual Nearest Neighbor) algorithm

    Args:
        cost_matrix (np.array): 3-ndim [N_cls, N_det, N_tra] or 2-ndim, invaild cost equal to np.inf
        thresholds (dict): matching thresholds to restrict FP matches

    Returns:
        Tuple[list, list, np.array, np.array]: matched det, matched tra, unmatched det, unmatched tra
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2: cost_matrix = cost_matrix[None, :, :]
    assert len(thresholds) == cost_matrix.shape[0], "the number of thresholds should be egual to cost matrix number."
    
    # solve cost matrix
    m_det, m_tra = [], []
    num_det, num_tra = cost_matrix.shape[1:] 
    for cls_idx, cls_cost in enumerate(cost_matrix):
        mask = cls_cost <= thresholds[cls_idx]
        # mutual nearest
        mask = mask \
            * (cls_cost == cls_cost.min(axis= 0)) \
            * (cls_cost == cls_cost.min(axis= 1)[:, np.newaxis])
        m_det += np.where(mask == True)[0].tolist()
        m_tra += np.where(mask == True)[1].tolist()
    
    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))
    return m_det, m_tra, um_det, um_tra