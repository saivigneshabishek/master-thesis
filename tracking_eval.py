# run me with environment=environment +experiment=eval_tracking
import os
import json
import time
import hydra
import wandb
from tqdm import tqdm
from tracking import MultiObjectTracker
from dataloader import create_dataloader
from omegaconf import DictConfig, OmegaConf
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs

@hydra.main(config_path="config", config_name="eval_config", version_base="1.3")
def main(cfg):


    cfg = OmegaConf.to_object(cfg)
    results_path = os.path.join(os.getcwd(), cfg['environment']['output_base_path'], 'tracking', cfg['run_name'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    # init wandb
    wandb.init(entity='ssm_pred',
                project='TRACKING',
                mode='online',
                name=cfg['run_name'],
                dir=results_path,
                config=cfg)
    
    dataloader = create_dataloader(cfg['dataloader'])

    tracker = MultiObjectTracker(cfg['tracking'])
    db = {"results":{}, "meta": {"use_camera": True, "use_lidar": True, "use_radar": False, "use_map": False, "use_external": False}}

    print('Starting to track....')
    start = time.time()
    for (inputs,_) in tqdm(dataloader):
        sample = inputs['sample']
        data = tracker.tracking(inputs)
        if data is not None:
            db['results'][sample[0]] = data

    db_path = os.path.join(results_path, 'results.json')
    with open(db_path, 'w') as f:
        json.dump(db, f)
    end = time.time()
    print(f'Completed in {(end-start)/60} minutes!')

    track_cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=track_cfg,
        result_path=db_path,
        eval_set=cfg['nusc']['eval_set'],
        output_dir=results_path,
        verbose=True,
        nusc_version=cfg['nusc']['version'],
        nusc_dataroot=cfg['nusc']['path'],
        )
    
    summary = nusc_eval.main()
    for item in summary.keys():
        if item in ['amota', 'ids', 'fp', 'tn', 'fn', 'amotp', 'frag']:
            wandb.log({f'tracking/{item}': summary[item]})

    wandb.finish()

if __name__ == "__main__":
    main()
