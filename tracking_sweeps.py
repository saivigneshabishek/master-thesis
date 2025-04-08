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

@hydra.main(config_path="config", config_name="evalsweep_config", version_base="1.3")
def main(cfg):
    # setup paths
    run = wandb.init()
    results_path = os.path.join(os.getcwd(), cfg['environment']['output_base_path'], 'tracking_sweeps', 'MAMBA_BASE_TRAIN_TRACK' , cfg['run_name'], run.name)
    
    # setup wandb
    cfg = OmegaConf.to_object(cfg)

    cfg['tracking']['track']['MAMBA_Tuning']['R_xyz'] = wandb.config.MAMBA_R_xyz
    cfg['tracking']['track']['MAMBA_Tuning']['R_wlh'] = wandb.config.MAMBA_R_wlh
    cfg['tracking']['track']['MAMBA_Tuning']['R_vxy'] = wandb.config.MAMBA_R_vxy
    cfg['tracking']['track']['MAMBA_Tuning']['R_r'] = wandb.config.MAMBA_R_r

    cfg['tracking']['track']['MAMBA_Tuning']['Q_xyz'] = wandb.config.MAMBA_Q_xyz
    cfg['tracking']['track']['MAMBA_Tuning']['Q_wlh'] = wandb.config.MAMBA_Q_wlh
    cfg['tracking']['track']['MAMBA_Tuning']['Q_vxy'] = wandb.config.MAMBA_Q_vxy
    cfg['tracking']['track']['MAMBA_Tuning']['Q_r'] = wandb.config.MAMBA_Q_r

    wandb.config = cfg

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

    if not os.path.exists(results_path):
        os.makedirs(results_path)
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
    
    summary = nusc_eval.main(render_curves=False)
    for item in summary.keys():
        if item in ['amota', 'ids', 'fp', 'tn', 'fn', 'amotp', 'frag']:
            wandb.log({f'tracking/{item}': summary[item]})

    wandb.finish()

if __name__ == "__main__":
    main()