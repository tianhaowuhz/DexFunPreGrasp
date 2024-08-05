python src/train.py \
    headless=False \
    env_mode=pgm \
    env_info=False \
    num_envs=12000 \
    num_objects=-1 \
    num_objects_per_env=1 \
    graphics_device_id=-1 \
    split='train' \
    cluster=0 \
    task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    task.env.datasetPoseLevelSampling=True \
    --seed=0 \
    --exp_name='PPO' \
    --logdir='pose_level_full_observation_cluster_0' \
    --run_device_id=0 \
    --web_visualizer_port=-1