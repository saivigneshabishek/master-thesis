### How to Use

Place the raw detection files (.json/.pkl) of the detectors in their respective folders.
- Detections for CenterPointLidar are downloaded from [here](https://mitprod-my.sharepoint.com/:f:/g/personal/tianweiy_mit_edu/Er_nsH9Z2tRHnptBFJ_ompAByE3zu4E88xae691xyS6q_w?e=UqTmU2) (With flip augmentation, 68.5 NDS val performance).
- TransFusionLidar and BEVFusion detections are generated using OpenPCDet Library.

### Format

Supports standard NuScenes Detection format.

```
submission {
    "meta": {
        "use_camera":   <bool>          -- Whether this submission uses camera data as an input.
        "use_lidar":    <bool>          -- Whether this submission uses lidar data as an input.
        "use_radar":    <bool>          -- Whether this submission uses radar data as an input.
        "use_map":      <bool>          -- Whether this submission uses map data as an input.
        "use_external": <bool>          -- Whether this submission uses external data as an input.
    },
    "results": {
        sample_token <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
}

```

```
sample_result {
    "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
    "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
    "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
    "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
    "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                           See table below for valid attributes for each class, e.g. cycle.with_rider.
                                           Attributes are ignored for classes without attributes.
                                           There are a few cases (0.4%) where attributes are missing also for classes
                                           that should have them. We ignore the predicted attributes for these cases.
}

```
