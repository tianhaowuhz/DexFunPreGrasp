import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--root_dir", type=str, default="/home/thwu/Projects/func-mani/results/demo")
    parser.add_argument("--method", type=str, default="diffusion")
    parser.add_argument("--num_workers_per_device", type=int, default=4)
    parser.add_argument("--num_envs_per_worker", type=int, default=1)
    parser.add_argument("--num_objects_per_worker", type=int, default=1)
    args = parser.parse_args()

    devices = [int(device) for device in args.device.split(",")]
    num_gpus = len(devices)
    num_workers_per_device = args.num_workers_per_device
    num_envs_per_worker = args.num_envs_per_worker
    num_objects_per_worker = args.num_objects_per_worker

    script_template = """python src/render_traj.py \
                    --demo_path={demo_path}  \
                    --method={method} \
                    --root_dir={root_dir} \
                    --run_device_id={run_device_id} \
                    """

    # multiple workers per device
    current_workers = {device: 0 for device in devices}

    def run_script(script):
        return os.system(script)

    processes = []

    data_path = os.path.join(args.root_dir, args.method)
    target_path = os.path.join(args.root_dir, "videos", args.method)

    total_traj_path = os.listdir(data_path)
    traj_running = np.zeros(len(total_traj_path))

    for video in os.listdir(target_path):
        if video.replace("mp4", "npy") in total_traj_path:
            video_index = total_traj_path.index(video.replace("mp4", "npy"))
            traj_running[video_index] = 1

    while not traj_running.all():
        for traj_id in np.where(traj_running == 0)[0]:
            traj_id = int(traj_id)
            traj = total_traj_path[traj_id]
            for device in devices:
                if current_workers[device] >= num_workers_per_device:
                    continue

                traj_running[traj_id] = 1
                current_script = script_template.format(
                    demo_path=traj, method=args.method, root_dir=args.root_dir, run_device_id=device
                )

                process = multiprocessing.Process(target=run_script, args=(current_script,))
                process.start()
                processes.append((process, device))

            for device in current_workers.keys():
                current_workers[device] = sum([1 for p, dev in processes if p.is_alive() and dev == device])

            processes = [(p, dev) for p, dev in processes if p.is_alive()]
