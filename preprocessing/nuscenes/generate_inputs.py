'''
splits raw detection json files into individual samples and converts them into protobuff objects 
'''

from tqdm import tqdm
import os
import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox
import nuscenes.utils.splits as splits
from utils import create_protobuf_objlist, create_protobuf_obj, write_protobuf, makedir, filter_dets
import numpy as np
from pyquaternion import Quaternion

def generate_inputs(dets, nusc, scenes, save_dir, split='train', ifego=False):

    print(f'--- Generating data for {split} split ----')
    for scene in tqdm(scenes):
        sample_token = scene['first_sample_token']
        last_token = scene['last_sample_token']
        last_sample = nusc.get('sample', last_token)
        # get egopose from the last frame
        sample_data_token = last_sample["data"]["LIDAR_TOP"]
        sd_record = nusc.get('sample_data', sample_data_token)
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        folder = os.path.join(save_dir, split, scene['name'])
        makedir(folder)

        # go through every sample in a scene sequentially
        while True:
            sample = nusc.get('sample', sample_token)
            sample_dets = dets[sample_token]
            sample_dets = filter_dets(sample_dets)
            num_objs = len(sample_dets)
            obj_list = create_protobuf_objlist(num_objs, sample['timestamp'])

            for detection in sample_dets:
                # global frame to ego frame
                # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L587
                if ifego:
                    global_to_ego_matrix = Quaternion(pose_record['rotation']).inverse
                    # translate
                    detection['translation'] += -np.array(pose_record['translation'])
                    # rotate
                    detection['translation'] = np.dot(global_to_ego_matrix.rotation_matrix, detection['translation'])
                    detection['velocity'] = np.dot(global_to_ego_matrix.rotation_matrix,
                                            [detection['velocity'][0], detection['velocity'][1], 0.0])[:2]
                    detection['rotation'] = (global_to_ego_matrix * Quaternion(detection['rotation'])).q

                create_protobuf_obj(obj_list, detection, type='detection')

            save_sample = os.path.join(folder, f'{sample_token}.bin')
            write_protobuf(obj_list, save_sample)

            if sample_token == scene['last_sample_token']:
                break
            else:
                sample_token = sample['next']

if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Process nuscenes raw detection results')
    args.add_argument('--data', required=True, type=str, help='Path to nuscenes dataset')
    args.add_argument('--detections', required=True, type=str, help='Path to raw detection json files')
    args.add_argument('--save', required=True, type=str, help='Path to save processed protobuf objects')
    args.add_argument('--version', required=True, type=str, help='Nuscenes version')
    args.add_argument('--egoframe', default=False, type=bool, help='Convert detections from global to ego frame')
    args = args.parse_args()
    assert args.version in ['v1.0-trainval', 'v1.0-mini', 'v1.0-test']

    nusc = NuScenes(version=args.version, dataroot=args.data, verbose=False)
    scenes = [scene for scene in nusc.scene]

    if args.version in ['v1.0-trainval', 'v1.0-mini']:
        if args.version == 'v1.0-trainval':
            train = [scene for scene in scenes if scene['name'] in splits.train]
            val = [scene for scene in scenes if scene['name'] in splits.val]
        else:
            train = [scene for scene in scenes if scene['name'] in splits.mini_train]
            val = [scene for scene in scenes if scene['name'] in splits.mini_val]
        
        # generate train and val detections objects
        train_detections = os.path.join(args.detections, 'train.json')
        dets, _ = load_prediction(train_detections, 1000, DetectionBox, verbose=False)
        generate_inputs(dets, nusc, train, args.save, split='train', ifego=args.egoframe)

        val_detections = os.path.join(args.detections, 'val.json')
        dets, _ = load_prediction(val_detections, 1000, DetectionBox, verbose=False)
        generate_inputs(dets, nusc, val, args.save, split='val', ifego=args.egoframe)

    else:
        ## todo: after downloading complete nuscenes dataset
        raise NotImplementedError