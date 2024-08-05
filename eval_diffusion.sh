################## train
# python src/run.py \
#     headless=False \
#     env_mode=pgm \
#     env_info=False \
#     num_envs=6968 \
#     num_objects=-1 \
#     num_objects_per_env=1 \
#     graphics_device_id=-1 \
#     split="train" \
#     task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
#     task.env.datasetPoseLevelSampling=True \
#     train.policy.network.n_layer=4 \
#     train.learn.dataset.data_dir="data/expert_dataset_pose_level_specialist/memmap" \
#     train.policy.network.encode_state_type="arm+dof+obj2palmpose+target" \
#     stack_frame_number=2 \
#     --algorithm="diffusion" \
#     --mode="eval" \
#     --exp_name="eval" \
#     --device_id=1 \
#     --num_evaluation_rounds=5 \
#     --print_freq=10 \
#     --checkpoint="ckpt/ddpm.pt"

################## test
python src/run.py \
    headless=False \
    env_mode=pgm \
    env_info=False \
    num_envs=3034 \
    num_objects=443 \
    num_objects_per_env=1 \
    graphics_device_id=-1 \
    split="test" \
    task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    task.env.datasetPoseLevelSampling=True \
    train.policy.network.n_layer=4 \
    train.learn.dataset.data_dir="data/expert_dataset_pose_level_specialist/memmap" \
    train.policy.network.encode_state_type="arm+dof+obj2palmpose+target" \
    stack_frame_number=2 \
    --algorithm="diffusion" \
    --mode="eval" \
    --exp_name="test" \
    --device_id=1 \
    --num_evaluation_rounds=5 \
    --print_freq=10 \
    --checkpoint="ckpt/ddpm.pt"
