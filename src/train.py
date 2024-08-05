import isaacgym
from hydra._internal.utils import get_args_parser
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from algorithms.ppo import PPO
from tasks import load_isaacgym_env
from utils.config import get_args, load_cfg
# from utils.vis import Visualizer # use visualizer requires to install sim-web-visualizer

if __name__ == "__main__":
    set_np_formatting()

    # argparse
    parser = get_args_parser()
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations to run")
    parser.add_argument("--seed", type=int, default=0, help="Seed Number")
    parser.add_argument("--run_device_id", type=int, default=0, help="Device id")

    parser.add_argument(
        "--torch_deterministic",
        action="store_true",
        default=False,
        help="Apply additional PyTorch settings for more deterministic behaviour",
    )
    parser.add_argument("--test", action="store_true", default=False, help="Run trained policy, no training")
    parser.add_argument("--con", action="store_true", default=False, help="whether continue train")
    parser.add_argument(
        "--web_visualizer_port", type=int, default=-1, help="port to visualize in web visualizer, set to -1 to disable"
    )
    parser.add_argument("--collect_demo_num", type=int, default=-1, help="collect demo num")
    parser.add_argument("--eval_times", type=int, default=5, help="Eval times for each object")
    parser.add_argument("--max_iterations", type=int, default=-1, help="Max iterations for training")

    parser.add_argument(
        "--cfg_train", type=str, default="ShadowHandFunctionalManipulationUnderarmPPO", help="Training config"
    )

    parser.add_argument("--logdir", type=str, default="", help="Log directory")
    parser.add_argument("--method", type=str, default="", help="Method name")
    parser.add_argument("--exp_name", type=str, default="", help="Exp name")
    parser.add_argument("--model_dir", type=str, default="", help="Choose a model dir")
    parser.add_argument("--eval_name", type=str, default="", help="Eval metric saving name")
    parser.add_argument("--vis_env_num", type=int, default=0, help="Number of env to visualize")

    # score matching parameter
    parser.add_argument("--t0", type=float, default=0.05, help="t0 for sample")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="num of hidden dim")
    parser.add_argument("--embed_dim", type=int, default=512, help="num of embed_dim")
    parser.add_argument("--score_mode", type=str, default="target", help="score mode")
    parser.add_argument("--space", type=str, default="euler", help="angle space")
    parser.add_argument("--cond_on_arm", action="store_true", help="dual score")
    parser.add_argument("--n_obs_steps", type=int, default=2, help="observation steps")
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--n_prediction_steps", type=int, default=4)
    parser.add_argument("--encode_state_type", type=str, default="all", help="encode state type")
    parser.add_argument(
        "--score_action_type",
        type=str,
        default="all",
        metavar="SCORE_ACTION_TYPE",
        help="score action type: arm, hand, all",
    )
    parser.add_argument(
        "--action_mode", type=str, default="rel", metavar="ACTION_MODE", help="action mode: rel, abs, obs"
    )
    parser.add_argument(
        "--score_model_path",
        type=str,
        default="/home/thwu/Projects/func-mani/ckpt/score_all.pt",
        help="pretrain score model path",
    )

    args = parser.parse_args()

    # if args.web_visualizer_port != -1:
    #     visualizer = Visualizer(args.web_visualizer_port)

    sim_device = f"cuda:{args.run_device_id}"
    rl_device = f"cuda:{args.run_device_id}"

    cfg_train, logdir = load_cfg(args)

    # set the seed for reproducibility
    set_seed(args.seed)
    """Change for different methods."""
    action_space = ["hand_rotation"]
    
    if args.exp_name == "PPO":
        if "env_mode=orn" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position",
                "fingertip_orientation",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_orn",
                "orientation_error",
            ]
        elif "env_mode=relpose" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relpose",
                "orientation_error",
                "position_error",
            ]
        elif "env_mode=relposecontact" in args.overrides:
            obs_space = [
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relposecontact",
                "orientation_error",
                "position_error",
                "fingerjoint_error",
                # "pointcloud_wrt_palm"
            ]
        elif "env_mode=pgm" in args.overrides:
            obs_space = [
                # "shadow_hand_position",
                # "shadow_hand_orientation",
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
                # "object_category",
                # "pointcloud_wrt_palm"
            ]
            action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
        # training parameter
        cfg_train["learn"]["nsteps"] = 8
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
        cfg_train["learn"]["clip_range"] = 0.1
    elif args.exp_name == "ppo_real":
        obs_space = [
            "ur_endeffector_position",
            "ur_endeffector_orientation",
            "shadow_hand_dof_position",
            "object_position_wrt_palm",
            "object_orientation_wrt_palm",
            "object_target_relposecontact",
            # "object_bbox",
        ]
        action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
        # training parameter
        cfg_train["learn"]["nsteps"] = 8
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
        cfg_train["learn"]["clip_range"] = 0.1
    else:
        raise NotImplementedError(f"setting {args.exp_name} not supported") 
    """
    load env
    """
    # override env args
    args.overrides.append(f"seed={args.seed}")
    args.overrides.append(f"sim_device={sim_device}")
    args.overrides.append(f"rl_device={rl_device}")
    args.overrides.append(f"obs_space={obs_space}")
    args.overrides.append(f"action_space={action_space}")
    # Load and wrap the Isaac Gym environment
    env = load_isaacgym_env(
        task_name="ShadowHandFunctionalManipulationUnderarm", args=args
    )  # preview 3 and 4 use the same loader
    # env = wrap_env(env)
    """
    load agent
    """
    learn_cfg = cfg_train["learn"]
    if "mode=eval" in args.overrides:
        learn_cfg["test"] = True
    is_testing = learn_cfg["test"]
    # Override resume and testing flags if they are passed as parameters.
    if args.model_dir != "":
        chkpt_path = args.model_dir

    runner = PPO(
        vec_env=env,
        cfg_train=cfg_train,
        device=rl_device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        asymmetric=False,
        args=args,
    )

    if args.model_dir != "":
        if is_testing:
            runner.test(chkpt_path)
        else:
            runner.load(chkpt_path)

    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    runner.run(num_learning_iterations=iterations, log_interval=cfg_train["learn"]["save_interval"])
