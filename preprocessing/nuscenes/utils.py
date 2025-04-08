import os
from evaluation_tools.class_definitions.perception import ObjectType, ObjectList
from google.protobuf.timestamp_pb2 import Timestamp
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

CLASSES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
# THRESHOLD = {
#     'bicycle': 0.15, 'bus':0 , 'car': 0.2, 'motorcycle': 0.2, 'pedestrian':0.2, 'trailer':0.1, 'truck':0.05,
#     }

THRESHOLD = {
    'bicycle': 0.14, 'bus':0 , 'car': 0.16, 'motorcycle': 0.16, 'pedestrian':0.16, 'trailer':0.1, 'truck':0,
    }

def convert_to_timestamp(time):
    ts = Timestamp()
    ts.seconds = time // 1000000
    ts.nanos = (time % 1000000) * 1000
    return ts

def create_protobuf_objlist(num_of_objects, time):
    obj_list =  ObjectList()
    obj_list.number_of_objects = num_of_objects
    # nuscene stores timestamp in microseconds, protobuff timestamps are in seconds, and fractions of a second as nanos
    timestamp = convert_to_timestamp(int(time))
    obj_list.timestamp.CopyFrom(timestamp)

    return obj_list

def create_protobuf_obj(obj_list, anno, type):
    assert type in ['detection', 'tracking'], "Wrong Object Type, supported object types are [detection, tracking]"

    det_obj = obj_list.objects.add()

    if type == 'detection':
        det_obj.type = ObjectType.OBJECT_TYPE_DETECTION
    else:
        det_obj.type = ObjectType.OBJECT_TYPE_TRACK
        det_obj.id = anno['tracking_id']
    
    det_obj.location3d.x, det_obj.location3d.y, det_obj.location3d.z = anno['translation']
    det_obj.size3d.width, det_obj.size3d.length, det_obj.size3d.height = anno['size']
    det_obj.rotation3d.w, det_obj.rotation3d.x, det_obj.rotation3d.y, det_obj.rotation3d.z = anno['rotation']
    det_obj.velocity2d.x , det_obj.velocity2d.y = anno['velocity']
    det_obj.existence_probability = anno['detection_score']
    cls = det_obj.classes.add()
    cls.name, cls.probability = anno['detection_name'], anno['detection_score']

    return det_obj

def filter_dets(dets):
    detections = []
    for det in dets:
        det = det.serialize()
        name = det['detection_name']
        if (name in CLASSES) and (det['detection_score'] > THRESHOLD[name]):
            detections.append(det)
    return detections

def write_protobuf(obj_list, path):
    with open(path, 'wb') as f:
        f.write(obj_list.SerializeToString())

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)