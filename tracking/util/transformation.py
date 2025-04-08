import numpy as np
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from tracking.util.utils import warp_to_pi

def global_to_ego(det, ego_data):
    '''Converts Detections/Tracks from GlobalCoord to EgoCoord frame
    det: [x, y, z, w, l, h, vx, vy, r]
    ego_data: contains ego_translation, egoPose and egoPoseInverse
    '''
    # ego_data is passed as None, when no transformation is needed
    if ego_data is None:
        return det
    
    yaw = det[-1]
    translation = det[:3]
    wlh = det[3:6]
    velocity = [det[6], det[7], 0.0]
    rotation = Quaternion(axis=(0, 0, 1), radians=yaw)

    # translate
    translation += -np.array(ego_data['ego_translation'])
    # rotate
    translation = np.dot(ego_data['to_ego'].rotation_matrix, translation)
    velocity = np.dot(ego_data['to_ego'].rotation_matrix, velocity)[:2]
    rotation = (ego_data['to_ego'] *rotation)
    yaw = warp_to_pi(quaternion_yaw(rotation))

    new_det = [translation[0], translation[1], translation[2],
                    wlh[0], wlh[1], wlh[2],
                    velocity[0], velocity[1], yaw]

    return new_det

def ego_to_global(track, ego_data):
    '''Converts Detections/Tracks from EgoCoord to GlobalCoord frame
    track: [x, y, z, w, l, h, vx, vy, r]
    ego_data: contains ego_translation, egoPose and egoPoseInverse
    '''
    # ego_data is passed as None, when no transformation is needed
    if ego_data is None:
        return track

    yaw = track[-1]
    translation = track[:3]
    wlh = track[3:6]
    velocity = [track[6], track[7], 0.0]
    rotation = Quaternion(axis=(0, 0, 1), radians=yaw)

    # rotate 
    translation = np.dot(ego_data['to_global'].rotation_matrix, translation)
    velocity = np.dot(ego_data['to_global'].rotation_matrix, velocity)[:2]
    rotation = (ego_data['to_global']*rotation)
    yaw = warp_to_pi(quaternion_yaw(rotation))

    # translate 
    translation += np.array(ego_data['ego_translation'])

    new_track = [translation[0], translation[1], translation[2],
                    wlh[0], wlh[1], wlh[2],
                    velocity[0], velocity[1], yaw]
    
    return new_track