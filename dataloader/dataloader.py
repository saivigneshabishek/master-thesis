import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from dataloader.utils.data_utils import load_json, read_object_list, proto2dict, proto2dict_ego, proto2dict_ego_aug
from dataloader.utils.cache import SharedCache
from nuscenes.nuscenes import NuScenes
import nuscenes.utils.splits as splits

class EvalSequential(Dataset):
    '''returns detections and targets with:
        - sample (sample token, can be set as name of frameinfo file for other datasets)
        - frame_id
        - dets : [x,y,z,w,l,h,vx,vy,r,cls,prob,id], shape (B,N,12) N to be padded if multiple batch
        - num_dets: N
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.split = 'val' if cfg['isVal'] else 'test'

        if self.split == 'val':
            self.info = load_json(os.path.join(cfg['path'], 'val.json'))
        elif self.split == 'test':
            self.info = load_json(os.path.join(cfg['path'], 'test.json'))
        else:
            raise NotImplementedError

        # self.split = 'train'
        self.input_path = os.path.join(cfg['path'], 'inputs', cfg['detector'], self.split)
        self.target_path = os.path.join(cfg['path'], 'labels', self.split)
        self.scenes = sorted(self.info['sequences'].keys())
        self.all_samples = self.generate_samples()
    
    def generate_samples(self):
        samples = []
        for scene_name in self.scenes:
            # if scene_name in splits.train_track:
            scene = self.info['sequences'][scene_name]
            nbr_samples = scene['nbr_samples']
            order = scene['sample_order']
            assert nbr_samples==len(order)

            for frame_id, sample in enumerate(order):
                samples.append({
                        'sample': sample,
                        'scene': scene_name,
                        'frame_id': frame_id,
                        'first_sample': order[0],
                        'last_sample': order[-1],
                        })
        return samples
    
    def load_data(self, data):
        curr_token = data['sample']
        first_token = data['first_sample']
        last_token = data['last_sample']
        inp = read_object_list(os.path.join(self.input_path, data['scene'], f'{curr_token}.bin'))
        out = read_object_list(os.path.join(self.target_path, data['scene'], f'{curr_token}.bin'))
        dets = proto2dict(inp, data['frame_id'], curr_token, first_token, last_token)
        targets = proto2dict(out, data['frame_id'], curr_token, first_token, last_token)
        return dets, targets
        
    def __getitem__(self, idx):
        current_sample = self.all_samples[idx]
        data, targets = self.load_data(current_sample)
        return data, targets

    def __len__(self):
        return len(self.all_samples)
   
class TrainMotionPrediction(Dataset):
    def __init__(self, cfg, isTrain=True):
        self.cfg = cfg
        self.split = 'train' if isTrain else 'val'

        if self.split == 'train':
            self.info = load_json(os.path.join(cfg['path'], 'train.json'))
        elif self.split == 'val':
            self.info = load_json(os.path.join(cfg['path'], 'val.json'))
        else:
            raise NotImplementedError

        # we use the gt tracks to train motion prediction task
        self.target_path = os.path.join(cfg['path'], 'labels', self.split)
        self.scenes = sorted(self.info['sequences'].keys())
        self.seq_len = cfg['seq_len']
        self.max_num = cfg['max_objs']
        #todo: add more checks here, eg:stride larger than entire seq
        self.stride = cfg['stride'] if cfg['stride'] >= 1 else 1
        self.device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("GPU not available, using CPU instead.")
        self.all_samples = self.generate_samples()

        # self.nusc = NuScenes(version=cfg['nusc']['version'], dataroot=cfg['nusc']['path'], verbose=False)
        self.augmentation = cfg['augmentation']

        self.cache = False
        self.cache_x = SharedCache(dims=(len(self.all_samples), self.seq_len, self.max_num, 9))
        self.cache_y = SharedCache(dims=(len(self.all_samples), self.seq_len, self.max_num, 9))
        self.cache_z = SharedCache(dims=(len(self.all_samples), self.seq_len, self.max_num, 9))
        
    def pad_objects(self, X,Y):
        ids_xlist = [x['dets'][:,-1] for x in X]
        ids_x = np.array(list(set.intersection(*map(set, ids_xlist))), dtype=np.float32)

        ids_ylist = [y['dets'][:,-1] for y in Y]
        ids_y = np.array(list(set.intersection(*map(set, ids_ylist))), dtype=np.float32)

        ids = np.intersect1d(ids_x, ids_y)
        num = len(ids)

        inputs , targets, mask = [], [], []

        for x,y in zip(X,Y):
            # 1.) remove objects that don't occur in all scenes
            x_ = np.zeros((self.max_num, 9), dtype=np.float32)
            y_ = np.zeros((self.max_num, 9), dtype=np.float32)
            z_ = np.zeros((self.max_num, 9), dtype=np.float32)

            if num != 0:
                x_[:num, :] = x['dets'][np.isin(x['dets'][:, -1], ids)][:,:9]
                y_[:num, :] = y['dets'][np.isin(y['dets'][:, -1], ids)][:,:9]
                z_[:num, :] = 1.0

                # temp soln
                x_ = np.nan_to_num(x_, nan=0.0)
                y_ = np.nan_to_num(y_, nan=0.0)
            
            inputs.append(x_), targets.append(y_), mask.append(z_)

        inputs, targets = torch.from_numpy(np.stack(inputs, axis=0)), torch.from_numpy(np.stack(targets, axis=0))
        mask = torch.from_numpy(np.stack(mask, axis=0))

        return inputs, targets, mask

    def generate_samples(self):
        samples = []
        for scene_name in self.scenes:
            nbr_samples = self.info['sequences'][scene_name]['nbr_samples']
            order = self.info['sequences'][scene_name]['sample_order']

            assert nbr_samples==len(order)
            # this method ignores the last elements in order when a strided sweep is not possible
            # find a better way to handle this...
            length = (nbr_samples-self.seq_len)// self.stride
            for i in range(length+1):
                idx = i*self.stride
                mini_x = order[:-1][idx:idx+self.seq_len]
                # targets
                mini_y = order[idx+1:idx+self.seq_len+1]

                # assert len(mini_x) == self.seq_len and len(mini_y) == self.seq_len, "mini_seq are of diff lengths"
                if len(mini_x) != self.seq_len or len(mini_y) != self.seq_len:
                    continue
                else:
                    samples.append({'scene': scene_name,
                                'first': order[0],
                                'mini_seq_x': mini_x,
                                'mini_seq_y': mini_y})
        return samples
    
    def load_data(self, data):
        inputs, targets = [], []
        # currently only returns input mini sequences [t1,t2,...,tseq_len] //
        for idx, (token_x, token_y) in enumerate(zip(data['mini_seq_x'], data['mini_seq_y'])):
            if self.augmentation and self.split == 'train':
                x = read_object_list(os.path.join(self.target_path, data['scene'], f'{token_x}.bin'))
                inp = proto2dict_ego_aug(x, idx, token_x, self.nusc, data['mini_seq_y'][-1])
                y = read_object_list(os.path.join(self.target_path, data['scene'], f'{token_y}.bin'))
                out = proto2dict_ego(y, idx, token_x, self.nusc, data['mini_seq_y'][-1])
            else:
                x = read_object_list(os.path.join(self.target_path, data['scene'], f'{token_x}.bin'))
                inp = proto2dict_ego(x, idx, token_x, self.nusc, data['mini_seq_y'][-1])
                y = read_object_list(os.path.join(self.target_path, data['scene'], f'{token_y}.bin'))
                out = proto2dict_ego(y, idx, token_x, self.nusc, data['mini_seq_y'][-1])
            inputs.append(inp), targets.append(out)

        return inputs, targets
    
    def __getitem__(self, idx):
        # targets to be handled here for motion prediction task
        # if not self.cache:
        #     current_data = self.all_samples[idx]
        #     inputs, targets = self.load_data(current_data)
        #     x,y,z = self.pad_objects(inputs, targets)
        #     self.cache_x._store(idx, x), self.cache_y._store(idx, y), self.cache_z._store(idx, z)

        #     # store values in disk to be directly read from and save system memory while running sweeps
        #     # save = {'x': x, 'y':y, 'z':z}
        #     # dir = os.path.join('/home/sai/2024-mt-deenadayalan/datasets/nuscenes/inputs/CenterPointLidarTransformStored/', self.split)
        #     # os.makedirs(dir, exist_ok=True)
        #     # name = os.path.join(dir, f'{idx}.pkl')
        #     # with open(name,'wb') as f:
        #     #     pickle.dump(save, f)

        #     return x,y,z
        # else:
        #     if self.augmentation and self.split == 'train':
        #         return self.cache_x._getaug(idx), self.cache_y._getaug(idx), self.cache_z._getaug(idx)
        #     else:
        #         return self.cache_x._get(idx), self.cache_y._get(idx), self.cache_z._get(idx) 

        path = os.path.join('/home/sai/2024-mt-deenadayalan/datasets/nuscenes/inputs/CenterPointLidarTransformStored/', self.split, f'{idx}.pkl')
        with open(path, 'rb') as f:
            data = pickle.load(f)
            x = data['x']
            y = data['y']
            z = data['z']
        return x, y, z

    def __len__(self):
        return len(self.all_samples)