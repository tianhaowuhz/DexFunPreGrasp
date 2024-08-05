import sys
from pathlib import Path

import isaacgym
from dotenv import find_dotenv

# import other modules
from hydra._internal.utils import get_args_parser

from algorithms.algo import Algorithm
from algorithms.DDIM import DiffusionPolicy
from tasks import create_env_from_config, parse_hydra_config

sys.path.append(str(Path(find_dotenv()).parent))

if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument(
        "--algorithm", type=str, default="behavior_cloning", help="algorithm name (default: behavior_cloning)"
    )
    parser.add_argument("--mode", type=str, default="eval", help="mode (default: eval)")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device id (default: 0)")
    parser.add_argument("--exp_name", type=str, default="run", help="experiment name (default: run)")
    parser.add_argument("--print_freq", type=int, default=5, help="print frequency (default: 10)")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint path (default: None)")
    parser.add_argument("--num_observation_steps", type=int, default=1, help="number of observation steps (default: 1)")
    parser.add_argument(
        "--num_evaluation_rounds", type=int, default=10, help="number of evaluation rounds (default: 10)"
    )
    args = parser.parse_args()

    assert args.mode in ["train", "eval"], f"mode {args.mode} not supported, use `train` or `eval`"
    training = args.mode == "train"
    device = f"cuda:{args.device_id}"
    experiment_name = args.exp_name

    observation_space = [
        "ur_endeffector_position",
        "ur_endeffector_orientation",
        "shadow_hand_dof_position",
        "shadow_hand_dof_velocity",
        "fingertip_position_wrt_palm",
        "fingertip_orientation_wrt_palm",
        "fingertip_linear_velocity",
        "fingertip_angular_velocity",
        "object_position_wrt_palm",
        "object_orientation_wrt_palm",
        "object_position",
        "object_orientation",
        "object_linear_velocity",
        "object_angular_velocity",
        "object_target_relposecontact",
        "position_error",
        "orientation_error",
        "fingerjoint_error",
        "object_bbox",
        # might can be comment for fast training
        "tactile",
        "object_pointcloud",
        "imagined_pointcloud",
    ]
    action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
    args.overrides.append(f"obs_space={observation_space}")
    args.overrides.append(f"action_space={action_space}")
    args.overrides.append(f"sim_device={device}")
    args.overrides.append(f"rl_device={device}")

    Runner: Algorithm
    if args.algorithm == "diffusion":
        args.overrides.append("train=ShadowHandFunctionalManipulationUnderarmDiffusion")
        Runner = DiffusionPolicy
    else:
        raise NotImplementedError(f"algorithm {args.algorithm} not supported, use `behavior_cloning` or `diffusion`")

    cfg = parse_hydra_config("ShadowHandFunctionalManipulationUnderarm", args=args)
    env = create_env_from_config(cfg)

    runner = Runner(
        env=env,
        cfg_train=cfg.train,
        experiment_name=experiment_name,
        device=device,
        training=training,
        checkpoint_path=args.checkpoint,
        print_freq=args.print_freq,
        num_observation_steps=args.num_observation_steps,
        num_evaluation_rounds=args.num_evaluation_rounds,
    )
    runner.run()
