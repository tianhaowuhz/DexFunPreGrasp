import argparse
import multiprocessing
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metainfo", type=str, default="data/oakink_filtered_metainfo.csv")
    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--num_workers_per_device", type=int, default=4)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)
    parser.add_argument("--num_objects_per_worker", type=int, default=1)
    parser.add_argument("--cluster", type=str, default="7,8,9,10,11,12,13,14,15,16,17,18,19")
    args = parser.parse_args()

    devices = [int(device) for device in args.device.split(",")]
    clusters = [int(cluster) for cluster in args.cluster.split(",")]
    num_gpus = len(devices)
    num_workers_per_device = args.num_workers_per_device
    num_envs_per_worker = args.num_envs_per_worker
    num_objects_per_worker = args.num_objects_per_worker

    metainfo = pd.read_csv(args.metainfo)
    metainfo = metainfo[metainfo["split"] == "test"]

    # script_template = """python src/run.py \
    # mode=eval \
    # headless=False \
    # env_mode=pgm \
    # env_info=False \
    # num_envs=2 \
    # num_objects=1 \
    # num_objects_per_env=1 \
    # graphics_device_id={device} \
    # split='test' \
    # task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    # task.env.datasetPoseLevelSampling=True \
    # task.env.renderTarget=True \
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # train.policy.model.backbone_partial.num_layers=4 \
    # +task.env.datasetQueries.pose={codes} \
    # task.task.randomization_params.actor_params.object_0.rigid_body_properties.mass.range="[0.1, 0.5]" \
    # --num_observation_steps=2 \
    # --algorithm=behavior_cloning \
    # --mode=eval \
    # --exp_name=video_bc \
    # --device_id={device} \
    # --num_evaluation_rounds=2 \
    # --print_freq=10 \
    # --checkpoint=logs/BehaviorCloning/02-27-16-21_behavior_cloning_specialist_wo_bbox_demo_10_layer_4_batch_size_x4_lr_x2/behavior_cloning_90_epochs.pt \
    # """

    script_template = """python src/run.py \
    mode=eval \
    headless=False \
    env_mode=pgm \
    env_info=False \
    num_envs=1 \
    num_objects=1 \
    num_objects_per_env=1 \
    graphics_device_id={device} \
    split='test' \
    task.env.datasetMetainfoPath="data/oakink_filtered_metainfo.csv" \
    task.env.datasetPoseLevelSampling=True \
    task.env.renderTarget=True \
    task.env.enableDebugVis=True \
    task.env.visEnvNum=1 \
    train.policy.network.n_layer=4 \
    train.policy.network.encode_state_type="arm+dof+obj2palmpose+target" \
    task.task.randomization_params.actor_params.object_0.rigid_body_properties.mass.range="[0.1, 0.5]" \
    +task.env.datasetQueries.pose={codes} \
    stack_frame_number=2 \
    --algorithm=diffusion \
    --mode=eval \
    --exp_name=video_diffusion \
    --device_id={device} \
    --num_evaluation_rounds=2 \
    --print_freq=10 \
    --checkpoint=logs/DiffusionPolicy/02-28-09-44_diffusion_specialist_wo_bbox_demo_10_layer_4_4xbz_2xlr/score_90.pt \
    """

    # multiple workers per device
    current_workers = {device: 0 for device in devices}

    def run_script(script):
        return os.system(script)

    processes = []

    for cluster in clusters:
        # codes = sorted(metainfo[metainfo['cluster'] == cluster]["code"].drop_duplicates().tolist())
        poses = sorted(metainfo[metainfo["cluster"] == cluster]["pose"].drop_duplicates().tolist())
        print(f"Cluster {cluster} has {len(poses)} poses.")
        current_start = 0
        while current_start < len(poses):
            for device in devices:
                if current_workers[device] >= num_workers_per_device:
                    continue

                logdir = f"cluster_{cluster}_start_{current_start}_device_{device}"
                current_script = script_template.format(
                    num_envs_per_worker=num_envs_per_worker,
                    num_objects_per_worker=num_objects_per_worker,
                    device=device,
                    cluster=cluster,
                    codes="[" + ",".join(poses[current_start : current_start + num_objects_per_worker]) + "]",
                    logdir=logdir,
                )

                process = multiprocessing.Process(target=run_script, args=(current_script,))
                process.start()
                processes.append((process, device))

                current_start += num_objects_per_worker

            for device in current_workers.keys():
                current_workers[device] = sum([1 for p, dev in processes if p.is_alive() and dev == device])

            processes = [(p, dev) for p, dev in processes if p.is_alive()]
