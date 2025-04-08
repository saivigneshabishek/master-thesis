## all functions are taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from typing import List, Union, Tuple
from pyquaternion import Quaternion

def mask_between_boxes(labels_a: np.array, labels_b: np.array) -> Union[np.array, np.array]:
    """
    :param labels_a: np.array, labels of a collection
    :param labels_b: np.array, labels of b collection
    :return: np.array[bool] np.array , mask matrix, 1 denotes different, 0 denotes same
    """
    mask = labels_a.reshape(-1, 1).repeat(len(labels_b), axis=1) != labels_b.reshape(1, -1).repeat(len(labels_a),
                                                                                                   axis=0)
    return mask, mask.reshape(-1)

def mask_tras_dets(cls_num, det_labels, tra_labels) -> np.array:
    """
    mask invalid cost between tras and dets
    :return: np.array[bool], [cls_num, det_num, tra_num], True denotes valid (det label == tra label == cls idx)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    cls_mask = np.ones(shape=(cls_num, det_num, tra_num)) * np.arange(cls_num)[:, None, None]
    # [det_num, tra_num], True denotes invalid(diff cls)
    same_mask, _ = mask_between_boxes(det_labels, tra_labels)
    # [det_num, tra_num], invalid idx assign -1
    tmp_labels = tra_labels[None, :].repeat(det_num, axis=0)
    tmp_labels[np.where(same_mask)] = -1
    return tmp_labels[None, :, :].repeat(cls_num, axis=0) == cls_mask

def spec_metric_mask(cls_list: list, det_labels: np.array, tra_labels: np.array) -> np.array:
    """
    mask matrix, merge all object instance index of the specific class
    :param cls_list: list, valid category list
    :param det_labels: np.array, class labels of detections
    :param tra_labels: np.array, class labels of trajectories
    :return: np.array[bool], True denotes invalid(the object's category is not specific)
    """
    det_num, tra_num = len(det_labels), len(tra_labels)
    metric_mask = np.ones((det_num, tra_num), dtype=bool)
    merge_det_idx = [idx for idx, cls in enumerate(det_labels) if cls in cls_list]
    merge_tra_idx = [idx for idx, cls in enumerate(tra_labels) if cls in cls_list]
    metric_mask[np.ix_(merge_det_idx, merge_tra_idx)] = False
    return metric_mask

def expand_dims(array: np.array, expand_len: int, dim: int) -> np.array:
    return np.expand_dims(array, dim).repeat(expand_len, axis=dim)

def logical_or_mask(mask: np.array, seq_mask: np.array, boxes_a: dict, boxes_b: dict) -> np.array:
    """
    merge all mask which True means invalid
    :param mask: np.array, mask matrix
    :param seq_mask: np.array, 1-d mask matrix
    :param boxes_a: dict, a boxes infos, keys may include 'mask'
    :param boxes_b: dict, b boxes infos, keys may include 'mask'
    :return: np.array, mask matrix after merging(logical or) all mask
    """
    if 'mask' in boxes_b or 'mask' in boxes_a:
        if 'mask' in boxes_b and 'mask' in boxes_a:
            mask_ab = np.logical_or(boxes_a['mask'], boxes_b['mask'])
        elif 'mask' in boxes_b:
            mask_ab = boxes_b['mask']
        elif 'mask' in boxes_a:
            mask_ab = boxes_a['mask']
        else: raise Exception("cannot be happened")
        mask = np.logical_or(mask, mask_ab)
        return mask, mask.reshape(-1)
    else:
        return mask, seq_mask
    
def loop_inter(polys1: List[Polygon], polys2: List[Polygon], mask: np.array) -> np.array:
    """
    :param polys1: List[Polygon], collection of polygons
    :param polys2: List[Polygon], collection of polygons
    :param mask: np.array[bool], True denotes Invalid, False denotes valid
    :return: np.array, intersection area between two polygon collections
    """
    inters = np.zeros_like(mask, float)
    for i, reca in enumerate(polys1):
        for j, recb in enumerate(polys2):
            inters[i, j] = reca.intersection(recb).area if not mask[i, j] else 0
    return inters


def loop_convex(bottom_corners_a: np.array, bottom_corners_b: np.array, mask: np.array) -> np.array:
    """
    :param bottom_corners_a: np.array, bottom corners of a polygons, [a_num, b_num, 4, 2]
    :param bottom_corners_b: np.array, bottom corners of b polygons, [a_num, b_num, 4, 2]
    :param mask: np.array[bool], True denotes Invalid, False denotes valid, [a_num, b_num]
    :return: np.array, convexhull areas between two polygons, [a_num, b_num]
    """

    def init_convex(bcs: np.array, mask_: np.array) -> np.array:
        fake_convex = ConvexHull(bcs[0])
        return [ConvexHull(bc) if not mask_[i] else fake_convex for i, bc in enumerate(bcs)]

    all_bcs = np.concatenate((bottom_corners_a, bottom_corners_b), axis=2).reshape(-1, 8, 2)  # [numa * numb, 8, 2]

    # construct convexhull for every two boxes
    convexs = init_convex(all_bcs, mask)
    conv_cors = np.array([convex.vertices for convex in convexs], dtype=object)

    # 9 denotes 9 possible situations of len(conv_cor)
    conv_nums = np.array([len(conv_cor) for conv_cor in conv_cors]).reshape(1, -1).repeat(9, axis=0)  # [9, numa * numb]
    poss_idxs = np.arange(9).reshape(-1, 1).repeat(len(conv_cors), axis=1)

    # True in each row means that the number of convexHull corner points is the same as corresponding row index
    idx_masks = (conv_nums == poss_idxs)
    row_valid_idx = [np.where(idx_mask) for idx_mask in idx_masks]

    # Obtain convexhull area in order of the points number
    convex_areas = np.zeros(len(conv_cors))
    for conv_num, valid_idx in enumerate(row_valid_idx):
        if len(valid_idx[0]) == 0: continue
        b_idx = np.arange(len(valid_idx[0])).reshape(-1, 1).repeat(conv_num, axis=1)  # [len(valid_idx), conv_num]
        i_idx = np.stack((np.array(cor, dtype=int) for cor in conv_cors[valid_idx]))  # [len(valid_idx), conv_num]
        convex_areas[valid_idx] = PolyArea2D(all_bcs[valid_idx][b_idx, i_idx, :])

    return convex_areas.reshape(bottom_corners_a.shape[:2])

def PolyArea2D(pts: np.array) -> np.array:
    """
    Parallel version for computing areas of polygons surrounded by pts
    :param pts: np.array, a collection of xy coordinates of points, [poly_num, pts_num, 2]
    :return: float, Areas of polygons surrounded by pts, [poly_num,]
    """
    roll_pts = np.roll(pts, -1, axis=1)
    area = np.abs(np.sum((pts[:, :, 0] * roll_pts[:, :, 1] - pts[:, :, 1] * roll_pts[:, :, 0]), axis=1)) * 0.5
    return area

def reorder_metrics(metrics: dict) -> dict:
    """
    reorder metrics from {key(class, int): value(metrics, str)} to {key(metrics, str): value(class_labels, list)}
    :param metrics: dict, format: {key(class, int): value(metrics, str)}
    :return: dict, {key(metrics, str): value(class_labels, list)}
    """
    new_metrics = {}
    for cls, metric in metrics.items():
        if metric in new_metrics: new_metrics[metric].append(cls)
        else: new_metrics[metric] = [cls]
    return new_metrics

def warp_to_pi(yaw: float) -> float:
    """warp yaw to [-pi, pi)

    Args:
        yaw (float): raw angle

    Returns:
        float: raw angle after warping
    """
    while yaw >= np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw

def corners(det) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = det[3:6]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    yaw = det[8]
    quat = Quaternion(axis=(0, 0, 1), radians=yaw)
    corners = np.dot(quat.rotation_matrix, corners)

    # Translate
    x, y, z = det[0:3]
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    corners = corners[:, [2, 3, 7, 6]]
    corners = corners[:2].T

    return corners

def box_corners(dets):
    bottom_corners = []
    for det in dets:
        bottom_corners.append(corners(det))
    return np.stack(bottom_corners, axis=0)

def stack_tracks(tracks, frame_id=None):
    ret = []
    for track in tracks.values():
        t = track.getLocalState(frame_id).tolist()
        add = [float(track.cls_id), float(track.score), float(track.id)]
        t.extend(add)
        ret.append(np.array(t))
    return np.stack(ret, axis=0)