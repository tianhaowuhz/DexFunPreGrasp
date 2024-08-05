# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import random
import sys

import numpy as np
import torch
import yaml
from isaacgym import gymapi, gymutil


def set_np_formatting():
    np.set_printoptions(
        edgeitems=30,
        infstr="inf",
        linewidth=4000,
        nanstr="nan",
        precision=2,
        suppress=False,
        threshold=10000,
        formatter=None,
    )


def warn_task_name():
    raise Exception("Unrecognized task!")


def warn_algorithm_name():
    raise Exception("Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args, use_rlg_config=False):
    if args.task in [
        "ShadowHandOver",
        "ShadowHandCatchUnderarm",
        "ShadowHandTwoCatchUnderarm",
        "ShadowHandCatchAbreast",
        "ShadowHandReOrientation",
        "ShadowHandCatchOver2Underarm",
        "ShadowHandBottleCap",
        "ShadowHandDoorCloseInward",
        "ShadowHandDoorCloseOutward",
        "ShadowHandDoorOpenInward",
        "ShadowHandDoorOpenOutward",
        "ShadowHandKettle",
        "ShadowHandPen",
        "ShadowHandSwitch",
        "ShadowHandPushBlock",
        "ShadowHandSwingCup",
        "ShadowHandGraspAndPlace",
        "ShadowHandScissors",
        "AllegroHandOver",
        "AllegroHandCatchUnderarm",
    ]:
        return (
            os.path.join(args.logdir, "{}/{}/{}".format(args.task, args.algo, args.algo)),
            "cfg/{}/config.yaml".format(args.algo),
            "cfg/{}.yaml".format(args.task),
        )

    elif args.task in ["ShadowHandLiftUnderarm"]:
        return (
            os.path.join(args.logdir, "{}/{}/{}".format(args.task, args.algo, args.algo)),
            "cfg/{}/lift_config.yaml".format(args.algo),
            "cfg/{}.yaml".format(args.task),
        )

    elif args.task in ["ShadowHandBlockStack"]:
        return (
            os.path.join(args.logdir, "{}/{}/{}".format(args.task, args.algo, args.algo)),
            "cfg/{}/stack_block_config.yaml".format(args.algo),
            "cfg/{}.yaml".format(args.task),
        )

    elif args.task in ["ShadowHand", "ShadowHandReOrientation"]:
        return (
            os.path.join(args.logdir, "{}/{}/{}".format(args.task, args.algo, args.algo)),
            "cfg/{}/re_orientation_config.yaml".format(args.algo),
            "cfg/{}.yaml".format(args.task),
        )

    else:
        warn_task_name()


def load_cfg(args):
    with open(os.path.join(os.path.dirname(__file__), "../../cfg/train/", args.cfg_train + ".yaml"), "r") as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    logdir = args.logdir

    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["torch_deterministic"] = True

    # Override seed if passed on the command line
    if args.seed is not None:
        cfg_train["seed"] = args.seed

    return cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False):
    custom_parameters = [
        # env
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:1",
            "help": "Choose CPU or GPU device for inferencing policy network",
        },
        {"name": "--randomize", "action": "store_true", "default": False, "help": "Apply physics domain randomization"},
        {
            "name": "--num_envs",
            "type": int,
            "default": 2,
            "help": "Number of environments to create - override config file",
        },
        {
            "name": "--episode_length",
            "type": int,
            "default": 0,
            "help": "Episode length, by default is read from yaml config",
        },
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--points_per_object", "type": int, "default": 1024, "help": "points for each object pcl"},
        {"name": "--method", "type": str, "default": "gf+rl", "help": "method"},
        {"name": "--run_device_id", "type": int, "help": "device id"},
        {"name": "--dataset_type", "type": str, "default": "train", "help": "method"},
        # mode
        {"name": "--mode", "type": str, "default": "train", "help": "env_mode"},
        {"name": "--test", "action": "store_true", "default": False, "help": "Run trained policy, no training"},
        {"name": "--eval_times", "type": int, "default": 5, "help": "eval times for each object"},
        {"name": "--constrained", "action": "store_true", "help": "whether constrain base"},
        # score matching parameter
        {"name": "--t0", "type": float, "default": 0.05, "help": "t0 for sample"},
        {"name": "--hidden_dim", "type": int, "default": 1024, "help": "num of hidden dim"},
        {"name": "--embed_dim", "type": int, "default": 512, "help": "num of embed_dim"},
        {"name": "--score_mode", "type": str, "default": "target", "help": "score mode"},
        {"name": "--space", "type": str, "default": "riemann", "help": "angle space"},
        {"name": "--encode_state_type", "type": str, "default": "all", "help": "encode state type"},
        {
            "name": "--score_model_path",
            "type": str,
            "default": "/home/thwu/Projects/func-mani/ckpt/score_all.pt",
            "help": "pretrain score model path",
        },
        # rl train
        {
            "name": "--torch_deterministic",
            "action": "store_true",
            "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour",
        },
        {
            "name": "--metadata",
            "action": "store_true",
            "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user",
        },
        {"name": "--resume", "type": int, "default": 0, "help": "Resume training or start testing from a checkpoint"},
        {"name": "--cfg_train", "type": str, "default": "ShadowHandFunctionalManipulationUnderarmPPO"},
        {"name": "--max_iterations", "type": int, "default": 0, "help": "Set a maximum number of training iterations"},
        {
            "name": "--minibatch_size",
            "type": int,
            "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings.",
        },
        # log
        {"name": "--logdir", "type": str, "default": "logs/gfppo/"},
        {
            "name": "--experiment",
            "type": str,
            "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name",
        },
        {"name": "--model_dir", "type": str, "default": "", "help": "Choose a model dir"},
        {"name": "--exp_name", "type": str, "default": "ours", "help": "exp_name"},
        {"name": "--eval_name", "type": str, "default": "ours", "help": "exp_name"},
        {"name": "--vis_env_num", "type": int, "default": "0", "help": "vis env num"},
    ]

    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else "cpu"

    if args.test:
        args.train = False
    else:
        args.train = True

    return args
