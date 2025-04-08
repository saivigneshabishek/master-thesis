'''
generates a file with info on scenes and samples, converts label annotations into protobuf objects
'''

from tqdm import tqdm
import os
import json
import argparse
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits
from nuscenes.eval.tracking.utils import category_to_tracking_name
from utils import create_protobuf_objlist, create_protobuf_obj, write_protobuf, makedir 
from pyquaternion import Quaternion
import numpy as np

def load_gt(nusc, sample):
    sample_anns_token = sample['anns']
    annotations = []
    for object_anns_token in sample_anns_token:
        anns = nusc.get('sample_annotation', object_anns_token)

        # Only keep classes relevant to tracking
        if category_to_tracking_name(anns['category_name']) is not None:
            
            obj_annotation = {
                'sample_token': sample['token'],
                'translation': np.array(anns['translation']),
                'size': anns['size'],
                'rotation': np.array(anns['rotation']),
                'velocity': nusc.box_velocity(anns['token'])[:2],
                'detection_name': category_to_tracking_name(anns['category_name']),
                'tracking_instance': anns['instance_token'],
                'tracking_id': -10,
                'detection_score': 1.0,
            }

            annotations.append(obj_annotation)

    return annotations

def generate_gt(nusc, scenes, save_dir, split='train', ifego=False):

    print(f'--- Generating ground truth labels for {split} split ----')
    db = {'sequences':{},
          'frequency': 2}
    for scene in tqdm(scenes):
        sample_token = scene['first_sample_token']
        folder = os.path.join(save_dir, split, scene['name'])
        makedir(folder)

        scene_anno = []
        sample_token_list = []
        time = []

        # go through every sample in a scene sequentially
        while True:
            sample_token_list.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_ann = load_gt(nusc, sample)
            scene_anno.append(sample_ann)
            time.append(sample['timestamp'])

            if sample_token == scene['last_sample_token']:
                # ego pose from the last frame in the scene
                sample_data_token = sample["data"]["LIDAR_TOP"]
                sd_record = nusc.get('sample_data', sample_data_token)
                pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
                break
            else:
                sample_token = sample['next']

        # set object id to be used for tracking
        assert len(scene_anno) == len(sample_token_list)
        tracking = []
        for sample_anno, token, timestamp in zip(scene_anno, sample_token_list, time):
            num_objs = len(sample_anno)
            obj_list = create_protobuf_objlist(num_objs, timestamp)

            for obj_anno in sample_anno:
                tracking_instance = obj_anno['tracking_instance']
                if tracking_instance not in tracking:
                    tracking.append(tracking_instance)
                tracking_id = tracking.index(tracking_instance)
                obj_anno.update({'tracking_id': tracking_id})

                # global frame to ego frame
                # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L587
                if ifego:
                    global_to_ego_matrix = Quaternion(pose_record['rotation']).inverse
                    
                    # translate
                    obj_anno['translation'] += -np.array(pose_record['translation'])
                    # rotate
                    obj_anno['translation'] = np.dot(global_to_ego_matrix.rotation_matrix, obj_anno['translation'])
                    obj_anno['velocity'] = np.dot(global_to_ego_matrix.rotation_matrix,
                                            [obj_anno['velocity'][0], obj_anno['velocity'][1], 0.0])[:2]
                    obj_anno['rotation'] = (global_to_ego_matrix * Quaternion(obj_anno['rotation'])).q

                # create tracking protobuff object
                create_protobuf_obj(obj_list, obj_anno, type='tracking')

            save_sample = os.path.join(folder, f'{token}.bin')
            write_protobuf(obj_list, save_sample)

        # info on scene and sample order
        scene_info = {
            scene['name']: {
                'token': scene['token'],
                'nbr_samples': scene['nbr_samples'],
                'sample_order': sample_token_list,
                'first_sample_token': scene['first_sample_token'],
                'last_sample_token': scene['last_sample_token']
            }
        }

    # db file containing info abt scenes
        db['sequences'].update(scene_info)
    infofile = os.path.join(save_dir, f'{split}.json')
    with open(infofile, 'w') as f:
        json.dump(db, f)


if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Process nuscenes annotations')
    args.add_argument('--data', required=True, type=str, help='Path to nuscenes dataset')
    args.add_argument('--save', required=False, type=str, help='Path to save processed protobuf objects')
    args.add_argument('--version', required=True, type=str, help='Nuscenes version')
    args.add_argument('--egoframe', default=False, type=bool, help='Convert tracks from global to ego frame')
    args = args.parse_args()
    assert args.version in ['v1.0-trainval', 'v1.0-mini']

    nusc = NuScenes(version=args.version, dataroot=args.data, verbose=False)
    scenes = [scene for scene in nusc.scene]

    if args.version == 'v1.0-trainval':
        train = [scene for scene in scenes if scene['name'] in splits.train]
        val = [scene for scene in scenes if scene['name'] in splits.val]
    else:
        train = [scene for scene in scenes if scene['name'] in splits.mini_train]
        val = [scene for scene in scenes if scene['name'] in splits.mini_val]
    
    generate_gt(nusc, train, args.save, split='train', ifego=args.egoframe)
    generate_gt(nusc, val, args.save, split='val', ifego=args.egoframe)