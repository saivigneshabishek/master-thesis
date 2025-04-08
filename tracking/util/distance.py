# fns to calculate distance metrics. 
# Taken from SimpleTrack (https://github.com/tusen-ai/SimpleTrack) and Poly-MOT (https://github.com/lixiaoyu2000/Poly-MOT)

import numpy as np
from typing import List, Tuple, Union
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tracking.util.utils import box_corners, mask_between_boxes, expand_dims, logical_or_mask, loop_convex, loop_inter

def giou_3d(boxes_a, boxes_b) -> Tuple[np.array, np.array]:
    """
    taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
    half-parallel implementation of 3d giou. why half? convexhull and intersection are still serial
    'boxes': np.array, [det_num, 12](x, y, z, w, l, h, vx, vy, r, cls_id, prob score, obj id)
    :return: [np.array, np.array], 3d giou/bev giou between two boxes collections
    """

    # load info
    bcs_a, bcs_b = box_corners(boxes_a), box_corners(boxes_b)   # [box_num, 4, 2]
    
    assert boxes_a.shape[1] == 12 and boxes_b.shape[1] == 12, "dims must be 12"
    infos_a = boxes_a # [box_num, 12]
    infos_b = boxes_b # [box_num, 12]

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 12 and infos_b.shape[1] == 12, "dims must be 12"

    # mask matrix, True denotes different(invalid), False denotes same(valid)
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -3], infos_b[:, -3])
    bool_mask, seq_mask = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    rep_bcs_a, rep_bcs_b = expand_dims(bcs_a, len(bcs_b), 1), expand_dims(bcs_b, len(bcs_a), 0)  # [a_num, b_num, 4, 2]
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    za, zb = expand_dims(infos_a[:, 2], len(infos_b), 1), expand_dims(infos_b[:, 2], len(infos_a), 0)  # [a_num, b_num]
    wa, la, ha = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_a[:, :, 2]
    wb, lb, hb = wlh_b[:, :, 0], wlh_b[:, :, 1], wlh_b[:, :, 2]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union height
    ohs = np.maximum(np.zeros_like(ha), np.minimum(za + ha / 2, zb + hb / 2) - np.maximum(za - ha / 2, zb - hb / 2))
    uhs = np.maximum((za + ha / 2), (zb + hb / 2)) - np.minimum((zb - hb / 2), (za - ha / 2))

    # overlap and union area/volume
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    inter_volumes = inter_areas * ohs
    union_areas, union_volumes = wa * la + wb * lb - inter_areas, wa * la * ha + wb * lb * hb - inter_volumes

    # convexhull area/volume
    convex_areas = loop_convex(rep_bcs_a, rep_bcs_b, seq_mask)
    convex_volumes = convex_areas * uhs

    # calu gioubev/giou3d and mask invalid value
    gioubev = inter_areas / union_areas - (convex_areas - union_areas) / convex_areas
    giou3d = inter_volumes / union_volumes - (convex_volumes - union_volumes) / convex_volumes
    giou3d[bool_mask], gioubev[bool_mask] = -np.inf, -np.inf

    return gioubev, giou3d

def iou_bev(boxes_a: dict, boxes_b: dict) -> np.array:
    """
    taken from Poly-MOT and modified to work here (https://github.com/lixiaoyu2000/Poly-MOT)
    half-parallel implementation of bev iou.
    'boxes': np.array, [det_num, 12](x, y, z, w, l, h, vx, vy, r, cls_id, prob score, obj id)
    :return: np.array, bev iou between two boxes collections
    """

    # load info
    bcs_a, bcs_b = box_corners(boxes_a), box_corners(boxes_b)   # [box_num, 4, 2]

    assert boxes_a.shape[1] == 12 and boxes_b.shape[1] == 12, "dims must be 12"
    infos_a = boxes_a # [box_num, 12]
    infos_b = boxes_b # [box_num, 12]

    # corner case, 1d array(collection only has one box) to 2d array
    if infos_a.ndim == 1: infos_a, bcs_a = infos_a[None, :], bcs_a[None, :]
    if infos_b.ndim == 1: infos_b, bcs_b = infos_b[None, :], bcs_b[None, :]
    assert infos_a.shape[1] == 12 and infos_b.shape[1] == 12, "dim must be 12"

    # mask matrix, True denotes different, False denotes same
    bool_mask, seq_mask = mask_between_boxes(infos_a[:, -3], infos_b[:, -3])
    bool_mask, _ = logical_or_mask(bool_mask, seq_mask, boxes_a, boxes_b)

    # process bottom corners, size and center for parallel computing
    wlh_a, wlh_b = expand_dims(infos_a[:, 3:6], len(infos_b), 1), expand_dims(infos_b[:, 3:6], len(infos_a), 0)
    wa, la, wb, lb = wlh_a[:, :, 0], wlh_a[:, :, 1], wlh_b[:, :, 0], wlh_b[:, :, 1]

    # polygons
    polys_a, polys_b = [Polygon(bc_a) for bc_a in bcs_a], [Polygon(bc_b) for bc_b in bcs_b]

    # overlap and union area
    inter_areas = loop_inter(polys_a, polys_b, bool_mask)
    union_areas = wa * la + wb * lb - inter_areas

    # calu bev iou and mask invalid value
    ioubev = inter_areas / union_areas
    ioubev[bool_mask] = -np.inf

    return ioubev

def box2corners2d(bbox):
        """ 
        taken from SimpleTrack (https://github.com/tusen-ai/SimpleTrack)
        the coordinates for bottom corners
        """
        x,y,z,w,l,h,o = bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5],bbox[8]
        bottom_center = np.array([x, y, z - h / 2])

        cos, sin = np.cos(o), np.sin(o)
        pc0 = np.array([x + cos * l / 2 + sin * w / 2,
                        y + sin * l / 2 - cos * w / 2,
                        z - h / 2])
        pc1 = np.array([x + cos * l / 2 - sin * w / 2,
                        y + sin * l / 2 + cos * w / 2,
                        z - h / 2])
        pc2 = 2 * bottom_center - pc0
        pc3 = 2 * bottom_center - pc1
    
        return [pc0.tolist(), pc1.tolist(), pc2.tolist(), pc3.tolist()]

def PolyArea2D(pts):
    '''taken from SimpleTrack (https://github.com/tusen-ai/SimpleTrack)'''
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def giou3d(box_a, box_b):
    '''taken from SimpleTrack (https://github.com/tusen-ai/SimpleTrack)'''
    boxa_corners = np.array(box2corners2d(box_a))[:, :2]
    boxb_corners = np.array(box2corners2d(box_b))[:, :2]
    reca, recb = Polygon(boxa_corners), Polygon(boxb_corners)
    ha, hb = box_a[5], box_b[5]
    za, zb = box_a[2], box_b[2]
    overlap_height = max(0, min([(za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2), ha, hb]))
    union_height = max([(za + ha / 2) - (zb - hb / 2), (zb + hb / 2) - (za - ha / 2), ha, hb])
    
    # compute intersection and union
    I = reca.intersection(recb).area * overlap_height
    U = box_a[3] * box_a[4] * ha + box_b[3] * box_b[4] * hb - I

    # compute the convex area
    all_corners = np.vstack((boxa_corners, boxb_corners))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)
    C = convex_area * union_height

    # compute giou
    giou = I / U - (C - U) / C
    return giou