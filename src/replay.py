import isaacgym
import numpy as np
import torch
from hydra._internal.utils import get_args_parser
from ipdb import set_trace
from isaacgym import gymapi
from isaacgymenvs.tasks.base import vec_task
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from sim_web_visualizer.isaac_visualizer_client import bind_visualizer_to_gym, create_isaac_visualizer, set_gpu_pipeline

from algorithms.ppo import PPO
from tasks import load_isaacgym_env
from utils.config import get_args, load_cfg


def wrapped_create_sim(
    self: vec_task.VecTask, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams
):
    sim = vec_task._create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    self.gym = bind_visualizer_to_gym(self.gym, sim)
    set_gpu_pipeline(sim_params.use_gpu_pipeline)
    return sim


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

    args = parser.parse_args()

    if args.web_visualizer_port != -1:
        # Reload VecTask function to create a hook for sim_web_visualizer
        vec_task.VecTask.create_sim = wrapped_create_sim
        # Create web visualizer
        create_isaac_visualizer(
            port=args.web_visualizer_port,
            host="localhost",
            keep_default_viewer=False,
            max_env=4,
            use_visual_material=False,
        )

    sim_device = f"cuda:{args.run_device_id}"
    rl_device = f"cuda:{args.run_device_id}"

    cfg_train, logdir = load_cfg(args)

    # set the seed for reproducibility
    set_seed(args.seed)

    demo_path = "data/expert_dataset/eval/demo_bottle_s118_0.npy"
    demo = np.load(demo_path, allow_pickle=True).tolist()
    obs_space = demo["obs_space"]
    action_space = demo["action_space"]
    predefined_object_codes = [demo["object_code"]]
    """Load env."""
    # override env args
    args.overrides.append(f"num_envs={1}")
    args.overrides.append(f"seed={args.seed}")
    args.overrides.append(f"sim_device={sim_device}")
    args.overrides.append(f"rl_device={rl_device}")
    args.overrides.append(f"obs_space={obs_space}")
    args.overrides.append(f"action_space={action_space}")
    args.overrides.append(f"predefined_object_codes={predefined_object_codes}")
    # Load and wrap the Isaac Gym environment
    env = load_isaacgym_env(
        task_name="ShadowHandFunctionalManipulationUnderarm", args=args
    )  # preview 3 and 4 use the same loader

    init_obj_pos = torch.tensor(demo["init_obj_pos"], dtype=torch.float32, device=env.device).reshape(1, -1)
    init_obj_orn = torch.tensor(demo["init_obj_orn"], dtype=torch.float32, device=env.device).reshape(1, -1)
    init_robot_dof = torch.tensor(demo["init_robot_dof"], dtype=torch.float32, device=env.device).reshape(1, -1)
    object_targets = torch.tensor(demo["target_dof"], dtype=torch.float32, device=env.device).reshape(1, -1)

    obs = torch.tensor(demo["obs"], dtype=torch.float32, device=env.device)
    action = torch.tensor(demo["action"], dtype=torch.float32, device=env.device)
    abs_action = torch.tensor(demo["abs_action"], dtype=torch.float32, device=env.device)

    env.set_states(robot_dof=init_robot_dof, object_targets=object_targets, obj_pos=init_obj_pos, obj_orn=init_obj_orn)
    for i in range(len(obs)):
        # test rel action
        # next_obs, rews, dones, infos = env.step(action[i : i + 1])

        # test abs action
        env.set_states(robot_dof=abs_action[i].unsqueeze(0), step_time=-1, set_dof_state=False, arm_ik=True)
        next_obs, rews, dones, infos = env.step(action[i : i + 1] * 0)

        # test obs action
        # env.set_states(robot_dof=obs[i,7:37].unsqueeze(0), step_time=-1, denomalize_robot_dof=True)
        # next_obs, rews, dones, infos = env.step(action[i : i + 1]*0)

        # test obs
        # env.set_states(
        #     robot_dof=obs[i, 7:37].unsqueeze(0),
        #     object_targets=object_targets,
        #     obj_pos=obs[i, 139:142].unsqueeze(0),
        #     obj_orn=obs[i, 142:146].unsqueeze(0),
        #     step_time=-1,
        #     denomalize_robot_dof=True,
        # )
        # next_obs, rews, dones, infos = env.step(action[i : i + 1] * 0)
    print(f"rot_dist:{env.rot_dist.item()} pos_dist:{env.pos_dist.item()} fj_dist:{env.fj_dist.item()}")
