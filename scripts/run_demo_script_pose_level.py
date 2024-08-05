import argparse
import multiprocessing
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metainfo", type=str, default="data/oakink_filtered_metainfo.csv")
    parser.add_argument("--device", type=str, default="0,1")
    parser.add_argument("--num_workers_per_device", type=int, default=4)
    parser.add_argument("--num_envs_per_worker", type=int, default=100)
    parser.add_argument("--num_objects_per_worker", type=int, default=5)
    parser.add_argument("--cluster", type=str, default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19")
    args = parser.parse_args()

    devices = [int(device) for device in args.device.split(",")]
    clusters = [int(cluster) for cluster in args.cluster.split(",")]
    num_gpus = len(devices)
    num_workers_per_device = args.num_workers_per_device
    num_envs_per_worker = args.num_envs_per_worker
    num_objects_per_worker = args.num_objects_per_worker

    metainfo = pd.read_csv(args.metainfo)
    metainfo = metainfo[metainfo["split"] == "train"]

    script_template = """python src/train.py \
        headless=False \
        env_mode=pgm \
        mode=eval \
        env_info=False \
        num_envs={num_envs_per_worker} \
        num_objects={num_objects_per_worker} \
        num_objects_per_env=1 \
        graphics_device_id=-1 \
        cluster={cluster} \
        +task.env.datasetQueries.pose={codes} \
        task.env.datasetPoseLevelSampling=True \
        --seed=0 \
        --exp_name='PPO' \
        --logdir={logdir} \
        --run_device_id={device} \
        --web_visualizer_port=-1 \
        --collect_demo_num=20 \
        --model_dir='ckpt/pose-level-specialist-{cluster}.pt'
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
