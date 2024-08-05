import argparse
import datetime
import glob
import os
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from utils.data import create_memmap_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments")
    parser.add_argument("--logging-dir", type=str, default="logs/PPO", help="Logging directory")
    parser.add_argument("--target-dir", type=str, default="data/expert_dataset", help="Target directory")
    parser.add_argument("--start-time", type=str, help="Start time for filtering trails")
    parser.add_argument("--end-time", type=str, help="End time for filtering trails")
    parser.add_argument("--force", action="store_true", help="Force removal of existing directory")
    parser.add_argument("--cluster", type=int, nargs="+", help="cluster id", default=None)
    args = parser.parse_args()

    cluster = list(range(20)) if args.cluster is None else args.cluster

    logging_dir = Path(args.logging_dir)
    target_dir = Path(args.target_dir)

    if target_dir.exists():
        print("target directory already exists")

        if not args.force:
            print("exiting")
            exit(0)
        else:
            print("removing existing directory")
            shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True)
    (target_dir / "train").mkdir()

    trails = glob.glob(str(logging_dir / "*"))
    trails = [Path(trail) for trail in trails]
    trails = [trail for trail in trails if trail.is_dir()]

    if args.start_time is not None:
        start_time = datetime.datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
        trails = [trail for trail in trails if trail.stat().st_mtime > start_time.timestamp()]
    if args.end_time is not None:
        end_time = datetime.datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
        trails = [trail for trail in trails if trail.stat().st_mtime < end_time.timestamp()]

    trails = sorted(trails, key=lambda x: x.stat().st_mtime)

    print(f"Number of trails: {len(trails)}")

    for i, trail in enumerate(tqdm(trails)):
        demonstration_dir = trail / "demo"
        if not demonstration_dir.exists():
            continue
        for demo in demonstration_dir.glob("*.npy"):
            cluster_id = int(trail.parts[2].split("_")[2])
            if cluster_id in cluster:
                # copy demonstration to target directory
                count = int(demo.name.split(".")[0].split("_")[-1])
                # if count <= 5:
                shutil.copy(demo, target_dir / "train" / f"{cluster_id}_{demo.name}")

    datadir = target_dir / "train"
    memmap_datadir = target_dir / "memmap"
    create_memmap_dataset(datadir, memmap_datadir)

    # remove raw directory
    shutil.rmtree(target_dir / "train")
