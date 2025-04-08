import torch
import numpy as np
import json
from evaluation_tools.class_definitions.perception import ObjectList, ObjectType
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

def numpy2tensor(data):
    """Transforms data to torch tensor"""
    return torch.from_numpy(data).float()

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def read_object_list(file_path):
    obj_list = ObjectList()
    with open(file_path, 'rb') as f:
        obj_list.ParseFromString(f.read())
    return obj_list

def nuscenes_class_encoding(obj_cls):
    CLASSES = {'bicycle':0, 'bus':1, 'car':2, 'motorcycle':3, 'pedestrian':4, 'trailer':5, 'truck':6}
    return CLASSES[obj_cls]

def proto2dict(objlist, frame_id, sample_token, first_token, last_token):
    dets = []
    for obj in objlist.objects:
        for cls in obj.classes:
            cls_id = np.array(nuscenes_class_encoding(cls.name), dtype=np.float32)
        
        # obj.type // 0 -> OBJECT_TYPE_UNKNOWN, 1 -> OBJECT_TYPE_DETECTION, 2-> OBJECT_TYPE_TRACK
        # if detection, set tracking_id to -1
        assert obj.type in [1,2]
        if obj.type == 2:
            obj_id = obj.id
        else:
            obj_id = -1.0

        #todo: check whether location, size, rotations are 2d/3d and then create np.array accordingly...
        # [x,y,z,w,l,h,vx,vy,r,cls,prob,id,ry(1x4)]
        quat = Quaternion([float(obj.rotation3d.w), float(obj.rotation3d.x),
                            float(obj.rotation3d.y), float(obj.rotation3d.z)])
        yaw = quaternion_yaw(quat)

        obj_det = np.array([obj.location3d.x, obj.location3d.y, obj.location3d.z,
                            obj.size3d.width, obj.size3d.length, obj.size3d.height,
                            obj.velocity2d.x, obj.velocity2d.y, yaw,
                            cls_id, obj.existence_probability, obj_id], dtype=np.float32)
        dets.append(obj_det)

    num = objlist.number_of_objects
    assert num == len(dets)
    ret = {
        'sample': sample_token,
        'first_sample': first_token,
        'last_sample': last_token,
        'frame_id': frame_id,
        'dets': np.stack(dets, axis=0) if len(dets) !=0 else np.ones((1,12))*-1,
        'num_dets': num,
    }

    return ret

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

def proto2dict_ego(objlist, frame_id, sample_token, nusc, last_token):
    dets = []
    sample = nusc.get('sample', last_token)
    sample_data_token = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for obj in objlist.objects:
        for cls in obj.classes:
            cls_id = np.array(nuscenes_class_encoding(cls.name), dtype=np.float32)
        
        # obj.type // 0 -> OBJECT_TYPE_UNKNOWN, 1 -> OBJECT_TYPE_DETECTION, 2-> OBJECT_TYPE_TRACK
        # if detection, set tracking_id to -1
        assert obj.type in [1,2]
        if obj.type == 2:
            obj_id = obj.id
        else:
            obj_id = -1.0

        translation = np.array([obj.location3d.x, obj.location3d.y, obj.location3d.z])
        velocity = np.array([obj.velocity2d.x, obj.velocity2d.y, 0.0])
        rotation = np.array([obj.rotation3d.w, obj.rotation3d.x, obj.rotation3d.y, obj.rotation3d.z])

        global_to_ego_matrix = Quaternion(pose_record['rotation']).inverse
        # translate
        translation += -np.array(pose_record['translation'])
        # rotate
        translation = np.dot(global_to_ego_matrix.rotation_matrix, translation)
        velocity = np.dot(global_to_ego_matrix.rotation_matrix, velocity)[:2]
        rotation = (global_to_ego_matrix * Quaternion(rotation)).q
        
        #todo: check whether location, size, rotations are 2d/3d and then create np.array accordingly...
        # [x,y,z,w,l,h,vx,vy,r,cls,prob,id,ry(1x4)]
        quat = Quaternion(rotation)
        yaw = quaternion_yaw(quat)
        yaw = warp_to_pi(quaternion_yaw(quat))

        obj_det = np.array([translation[0], translation[1], translation[2],
                            obj.size3d.width, obj.size3d.length, obj.size3d.height,
                            velocity[0], velocity[1], yaw,
                            cls_id, obj.existence_probability, obj_id], dtype=np.float32)
        dets.append(obj_det)

    num = objlist.number_of_objects
    assert num == len(dets)
    ret = {
        'sample': sample_token,
        'frame_id': frame_id,
        'dets': np.stack(dets, axis=0) if len(dets) !=0 else np.ones((1,12))*-1,
        'num_dets': num,
    }

    return ret

def proto2dict_ego_aug(objlist, frame_id, sample_token, nusc, last_token):
    dets = []
    sample = nusc.get('sample', last_token)
    sample_data_token = sample["data"]["LIDAR_TOP"]
    sd_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    for obj in objlist.objects:
        for cls in obj.classes:
            cls_id = np.array(nuscenes_class_encoding(cls.name), dtype=np.float32)
        
        # obj.type // 0 -> OBJECT_TYPE_UNKNOWN, 1 -> OBJECT_TYPE_DETECTION, 2-> OBJECT_TYPE_TRACK
        # if detection, set tracking_id to -1
        assert obj.type in [1,2]
        if obj.type == 2:
            obj_id = obj.id
        else:
            obj_id = -1.0

        translation = np.array([obj.location3d.x, obj.location3d.y, obj.location3d.z])
        trans_noise = np.random.normal([0,0,0], [606.35864*0.001, 416.74332*0.001, 0.7610718*0.001], 3)
        translation += trans_noise
        velocity = np.array([obj.velocity2d.x, obj.velocity2d.y, 0.0])
        vel_noise = np.random.normal([0,0], [2.352103*0.005, 2.0315256*0.005], 2)
        vel_noise = [vel_noise[0], vel_noise[1], 0.0]
        velocity += vel_noise
        rotation = np.array([obj.rotation3d.w, obj.rotation3d.x, obj.rotation3d.y, obj.rotation3d.z])

        global_to_ego_matrix = Quaternion(pose_record['rotation']).inverse
        # translate
        translation += -np.array(pose_record['translation'])
        # rotate
        translation = np.dot(global_to_ego_matrix.rotation_matrix, translation)
        velocity = np.dot(global_to_ego_matrix.rotation_matrix, velocity)[:2]
        rotation = (global_to_ego_matrix * Quaternion(rotation)).q
        
        #todo: check whether location, size, rotations are 2d/3d and then create np.array accordingly...
        # [x,y,z,w,l,h,vx,vy,r,cls,prob,id,ry(1x4)]
        quat = Quaternion(rotation)
        yaw = quaternion_yaw(quat)
        yaw_noise = np.random.normal(0, 1.8085531*0.001, 1)
        yaw = warp_to_pi(quaternion_yaw(quat))
        yaw += yaw_noise[0]

        extend = [obj.size3d.width, obj.size3d.length, obj.size3d.height]
        extend_noise = np.random.normal([0,0,0], [0.7295953*0.002, 2.8371725*0.002, 0.6377413*0.002], 3)
        extend += extend_noise

        obj_det = np.array([translation[0], translation[1], translation[2],
                            extend[0], extend[1], extend[2],
                            velocity[0], velocity[1], yaw,
                            cls_id, obj.existence_probability, obj_id], dtype=np.float32)
        dets.append(obj_det)

    num = objlist.number_of_objects
    assert num == len(dets)
    ret = {
        'sample': sample_token,
        'frame_id': frame_id,
        'dets': np.stack(dets, axis=0) if len(dets) !=0 else np.ones((1,12))*-1,
        'num_dets': num,
    }

    return ret

def nestedlen(array):
    length = 0
    for miniarray in array:
        if isinstance(miniarray, list):
            len = nestedlen(miniarray)
        else:
            len = 1
        length+=len
    return length