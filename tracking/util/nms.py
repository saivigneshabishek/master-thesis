"""
Non-Maximum Suppression(NMS) ops for the NuScenes dataset
"""

import numpy as np
import tracking.util.distance as metric_distances

METRIC = ['iou_3d', 'giou_3d']
def blend_nms(infos, cfg):
    """
    :param box_infos: dict, a collection of NuscBox info, keys must contain 'np_dets' and 'np_dets_bottom_corners'
    :param metrics: str, similarity metric for nms, five implemented metrics(iou_bev, iou_3d, giou_bev, giou_3d, d_eluc)
    :param thre: float, threshold of filter
    :return: keep box index, List[int]
    """
    metric = cfg['metric']
    threshold = cfg['threshold']
    assert metric in ['iou_bev', 'iou_3d', 'giou_bev', 'giou_3d', 'd_eucl'], "unsupported NMS metrics"

    sort_idxs, keep = np.argsort(-infos[:, -2]), []
    while sort_idxs.size > 0:
        i = sort_idxs[0]
        keep.append(i)
        # only one box left
        if sort_idxs.size == 1: break
        left, first = [infos[idx] for idx in [sort_idxs[1:], i]]
        first = first.reshape(1, -1)
        # the return value number varies by distinct metrics
        if metric not in METRIC: distances = getattr(metric_distances, metric)(first, left)[0]
        else: distances = getattr(metric_distances, metric)(first, left)[1][0]
        sort_idxs = sort_idxs[1:][distances <= threshold]

    return infos[keep]