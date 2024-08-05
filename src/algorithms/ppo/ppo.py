import copy
import functools
import glob
import io
import os
import pickle
import statistics
import time
from collections import deque
from datetime import datetime

import _pickle as CPickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gym.spaces import Space
from ipdb import set_trace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms.ppo.storage import RolloutStorage
# from algorithms.SDE import init_sde
# from networks.SDENets_update import CondScoreModel
from tasks.torch_utils import get_euler_xyz

from ..common.actor_critic import ActorCritic

save_video = False
img_size = 256
save_traj = False
ana = False
save_state = False
save_metric = False
obs_state_dim = 208
plot_direction = False
pcl_number = 512


def images_to_video(path, images, fps=10, size=(256, 256), suffix="mp4"):
    path = path + f".{suffix}"
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item.astype(np.uint8))
    out.release()


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class PPO:
    def __init__(
        self,
        vec_env,
        cfg_train,
        device="cpu",
        sampler="sequential",
        log_dir="",
        is_testing=False,
        print_log=True,
        apply_reset=False,
        asymmetric=False,
        args=None,
    ):
        self.args = args
        """PPO."""
        # PPO parameters
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cfg_train = copy.deepcopy(cfg_train)
        learn_cfg = self.cfg_train["learn"]
        self.device = device
        self.asymmetric = asymmetric
        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.num_transitions_per_env = learn_cfg["nsteps"]
        self.learning_rate = learn_cfg["optim_stepsize"]

        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)

        # policy type
        self.action_type = self.cfg_train["setting"]["action_type"]
        self.sub_action_type = self.cfg_train["setting"]["sub_action_type"]
        self.action_clip = self.cfg_train["setting"]["action_clip"]
        self.grad_process = self.cfg_train["setting"]["grad_process"]

        if self.action_type == "joint":
            if self.sub_action_type == "add+jointscale":
                action_space_shape = (vec_env.num_actions * 2,)
            elif self.sub_action_type == "addscale+add":
                action_space_shape = (vec_env.num_actions * 2 + 1,)
        else:
            action_space_shape = self.action_space.shape

        if "gf" in vec_env.observation_info:
            observation_space_shape = (int(self.observation_space.shape[0] / vec_env.stack_frame_number),)
        else:
            observation_space_shape = self.observation_space.shape

        self.vec_env = vec_env

        pointnet_version = self.cfg_train["policy"]["pointnet_version"]
        hand_pcl = self.cfg_train["policy"]["hand_pcl"]
        hand_model = None

        # obs_space = [
        #     # "shadow_hand_position",
        #     # "shadow_hand_orientation",
        #     "ur_endeffector_position",
        #     "ur_endeffector_orientation",
        #     "shadow_hand_dof_position",
        #     "shadow_hand_dof_velocity",
        #     "fingertip_position_wrt_palm",
        #     "fingertip_orientation_wrt_palm",
        #     "fingertip_linear_velocity",
        #     "fingertip_angular_velocity",
        #     "object_position_wrt_palm",
        #     "object_orientation_wrt_palm",
        #     "object_position",
        #     "object_orientation",
        #     "object_linear_velocity",
        #     "object_angular_velocity",
        #     "object_target_relposecontact",
        #     "position_error",
        #     "orientation_error",
        #     "fingerjoint_error",
        #     "object_bbox",
        #     # "object_category",
        #     # "pointcloud_wrt_palm"
        # ]
        # observation_metainfo = self.vec_env.export_observation_metainfo()
        # observation_metainfo = [obs for obs in observation_metainfo if obs['name'] in obs_space]
        # PPO components
        self.actor_critic = ActorCritic(
            observation_space_shape,
            self.state_space.shape,
            action_space_shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=asymmetric,
            pointnet_type=pointnet_version,
            observation_info=self.vec_env.export_observation_metainfo(),
            # observation_info=observation_metainfo,
            hand_pcl=hand_pcl,
            hand_model=hand_model,
            in_pointnet_feature_dim=4,  # TODO
            args=args,
            stack_frame_number=self.vec_env.stack_frame_number,
        )

        # pointnet backbone
        if self.actor_critic.pcl_dim > 0:
            self.pointnet_finetune = self.model_cfg["finetune_pointnet"]
            self.finetune_pointnet_bz = 128
            if self.model_cfg["pretrain_pointnet"]:
                if pointnet_version == "pt2":
                    pointnet_model_dict = torch.load(
                        os.path.join(args.score_model_path, "pointnet2.pt"), map_location=self.device
                    )
                elif pointnet_version == "pt":
                    pointnet_model_dict = torch.load(
                        os.path.join(args.score_model_path, "pointnet.pt"), map_location=self.device
                    )
                if self.model_cfg["shared_pointnet"]:
                    self.actor_critic.pointnet_enc.load_state_dict(pointnet_model_dict)
                    if not self.model_cfg["finetune_pointnet"]:
                        # freeze pointnet
                        for name, param in self.actor_critic.pointnet_enc.named_parameters():
                            param.requires_grad = False
                else:
                    self.actor_critic.actor_pointnet_enc.load_state_dict(pointnet_model_dict)
                    self.actor_critic.critic_pointnet_enc.load_state_dict(pointnet_model_dict)

                    if not self.model_cfg["finetune_pointnet"]:
                        # freeze pointnet
                        for name, param in self.actor_critic.actor_pointnet_enc.named_parameters():
                            param.requires_grad = False
                        for name, param in self.actor_critic.critic_pointnet_enc.named_parameters():
                            param.requires_grad = False

        self.actor_critic.to(self.device)
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            self.num_transitions_per_env,
            observation_space_shape,
            self.state_space.shape,
            action_space_shape,
            self.device,
            sampler,
        )

        if self.args.exp_name == "ilad":
            for name, param in self.actor_critic.additional_critic_mlp1.named_parameters():
                param.requires_grad = False

        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.learning_rate
        )
        """SDE."""
        if "gf" in self.vec_env.observation_info:
            # init SDE config
            prior_fn, marginal_prob_fn, sde_fn = init_sde("vp")

            # magic number
            self.score = CondScoreModel(
                marginal_prob_fn,
                hidden_dim=args.hidden_dim,
                embed_dim=args.embed_dim,
                mode=args.score_mode,
                pointnet_network_type="ori",
                action_dim=self.vec_env.num_actions,
                horizon=args.n_prediction_steps,
                obs_state_dim=obs_state_dim,
                obs_pcl_dim=pcl_number * 3,
                obs_horizon=self.vec_env.stack_frame_number,
                space=args.space,
                action_type=self.args.score_action_type,
                encode_state_type=args.encode_state_type,
                arm_action_dim=len(self.vec_env.ur_actuated_dof_indices),
                args=args,
            )
            if args.score_model_path == "":
                print("No score model found!")
            else:
                model_dict = torch.load(os.path.join(args.score_model_path, "score.pt"))
                self.score.load_state_dict(model_dict)
            self.score.to(device)
            self.score.eval()

            self.t0 = args.t0
            self.current_action = (
                torch.zeros(self.vec_env.num_envs, self.args.n_prediction_steps, self.vec_env.num_actions)
                .to(device)
                .float()
            )

        """
        ILAD
        """
        if self.args.exp_name == "ilad":
            from utils.data import IladMemmapTrajectoriesDataset

            self.lambda0 = 1.0
            self.lambda1 = 0.95
            self.lambda2 = 0.01
            self.lambda3 = 0.99
            if self.vec_env.mode == "train":
                """Load demo trajectories."""
                demo_data_dir = "/root/projects/func-mani/data/expert_dataset_synthetic_all/memmap"
                self.dataset = IladMemmapTrajectoriesDataset(
                    demo_data_dir, obs_info=self.vec_env.export_observation_metainfo(), squeeze_output=True
                )
                """Bahavior cloning."""
                self.bc_loss = nn.MSELoss()
                self.bc_optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.actor_critic.parameters()), lr=self.learning_rate
                )
                self.bc_bz = 256
                self.bc_epoch = 30

            if self.vec_env.mode == "train":
                dataloader = DataLoader(self.dataset, batch_size=self.bc_bz, shuffle=True, num_workers=4)

                for epoch in range(self.bc_epoch):
                    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                        observations, actions = data
                        observations = observations.to(self.device)
                        actions = actions.to(self.device)

                        preds = self.actor_critic.forward_actor(observations)

                        bc_loss = self.bc_loss(preds, actions)

                        self.bc_optimizer.zero_grad()
                        bc_loss.backward()
                        self.bc_optimizer.step()

                    print("epoch: {}, bc_loss: {}".format(epoch, bc_loss.item()))

                for name, param in self.actor_critic.additional_critic_mlp1.named_parameters():
                    param.requires_grad = True
                self.additional_critic_optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, self.actor_critic.additional_critic_mlp1.parameters()),
                    lr=self.learning_rate,
                )

        """Log."""
        # self.log_dir = log_dir
        if self.args.model_dir != "" and self.vec_env.mode == "train":
            time_now = self.args.model_dir.split("/")[-1].split("_")[0]
        else:
            time_now = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))

        if len(self.vec_env.object_codes) > 1:
            object_type = "all"
        else:
            object_type = self.vec_env.object_codes[0]

        if len(self.vec_env.label_paths) > 1:
            label_type = "all"
        else:
            label_type = self.vec_env.label_paths[0]

        if log_dir == "":
            pass
            # self.log_dir = os.path.join(
            #     f"./logs/{args.exp_name}/{time_now}_envnum:{self.vec_env.num_envs}_objnum:{self.vec_env.num_objects}_objnume:{self.vec_env.num_objects_per_env}_envmode:{self.vec_env.env_mode}_tranrewscale:{self.vec_env.tran_reward_scale}_seed{args.seed}"
            # )
        else:
            self.log_dir = f"./logs/{args.exp_name}/{time_now}_{log_dir}_objtype:{object_type}_labeltype:{label_type}_objnum:{self.vec_env.num_objects}_objcat:{self.vec_env.object_cat}_maxpercat:{self.vec_env.max_per_cat}_geo:{self.vec_env.object_geo_level}_scale:{self.vec_env.object_scale}_envnum:{self.vec_env.num_envs}_rewtype:{self.vec_env.reward_type}_seed{args.seed}"

            # tensorboard logging
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            # env cfg logging
            with open(os.path.join(self.log_dir, f"{self.vec_env.cfg['name']}.yaml"), "w") as f:
                OmegaConf.save(self.vec_env.cfg, f)
            # train cfg logging
            with open(os.path.join(self.log_dir, f"{args.cfg_train}.yaml"), "w") as f:
                yaml.dump(cfg_train, f)

        self.print_log = print_log

        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        if save_video:
            self.video_log_dir = os.path.join(self.log_dir, "video")
            os.makedirs(self.video_log_dir, exist_ok=True)
            self.vis_env_num = self.args.vis_env_num

        self.apply_reset = apply_reset
        """Evaluation."""
        self.eval_round = 2

        if self.vec_env.mode == "eval":
            self.eval_round = self.args.eval_times

        if save_metric:
            if save_state:
                self.eval_metrics = {
                    "obj_shapes": self.vec_env.object_codes,
                    "time_step": [],
                    "success_rate": [],
                    "gt_dist": [],
                    "stability": [],
                    "lift_nums": np.zeros(len(self.vec_env.object_codes)),
                    "gf_state_init": [],
                    "gf_state_final": [],
                    "gf_state_gt": [],
                }
            else:
                self.eval_metrics = {
                    "obj_shapes": self.vec_env.object_codes,
                    "time_step": [],
                    "success_rate": [],
                    "success_nums": np.zeros(len(self.vec_env.object_codes)),
                    "num_trials": np.zeros(len(self.vec_env.object_codes)),
                    "grasp": [],
                    "num_trails_per_grasp": [],
                    "num_success_per_grasp": [],
                }

            if self.vec_env.env_mode == "relpose":
                self.eval_metrics["pos_dist"] = []
                self.eval_metrics["rot_dist"] = []

        """ Demo """
        if self.args.collect_demo_num > 0:
            self.demo_dir = os.path.join(self.log_dir, "demo")
            os.makedirs(self.demo_dir, exist_ok=True)

            self.total_demo_num = self.args.collect_demo_num * len(self.vec_env.object_codes)
            self.cur_demo_num = 0

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        if self.args.con:
            self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()

        if save_metric:
            model_dir = path[: -len(path.split("/")[-1])] + f"metric_{self.args.exp_name}_{self.args.seed}.pkl"
            self.eval_metrics = CPickle.load(open(model_dir, "rb"))

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def get_action(self, current_obs, mode):
        # Compute the action
        actions, grad, _ = self.compute_action(current_obs=current_obs, mode=mode)
        step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())
        return step_actions

    def eval(self, it):
        # eval initilization
        self.vec_env.eval(vis=save_video)
        test_times = 0
        success_rates = []  # s_rate for each round
        reward_all = []
        eps_len_all = []
        succ_eps_len_all = []
        # demo_per_obj = {}
        demo_per_grasp = {}

        if self.vec_env.mode == "train":
            save_time = 0  # means save all videos
        else:
            save_time = self.eval_round - 1

        if self.args.collect_demo_num > 0:
            breakout_threshold = 10
            counter = 0
            while self.cur_demo_num < self.total_demo_num:
                counter += 1
                if counter > breakout_threshold:
                    break
                print(f"Current Round {counter}")
                print(f"# Demos: {self.cur_demo_num} / {self.total_demo_num}")
                # reset env
                self.vec_env.reset_arm()
                # TODO since reset not step simulation, the current obs is actually not correct
                current_obs = self.vec_env.reset()["obs"]
                print("reset env")
                eval_done_envs = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)

                self.demo_init_state = {}
                self.demo_init_state["obj_pos"] = self.vec_env.occupied_object_init_root_positions.cpu().numpy()
                self.demo_init_state["obj_orn"] = self.vec_env.occupied_object_init_root_orientations.cpu().numpy()
                self.demo_init_state["robot_dof"] = self.vec_env.robot_init_dof.cpu().numpy()
                self.demo_init_state["target_dof"] = self.vec_env.object_targets.cpu().numpy()
                self.demo_obs = torch.tensor([], device="cpu")
                self.demo_action = torch.tensor([], device="cpu")
                # self.demo_abs_action = torch.tensor([], device=self.device)
                # self.demo_obs_action = torch.tensor([], device=self.device)

                # step
                with torch.no_grad():
                    while True:
                        # Compute the action
                        actions, grad, _ = self.compute_action(current_obs=current_obs, mode="eval")
                        step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())

                        # Step the vec_environment
                        # print(step_actions)
                        # set done env action to be zero
                        done_env_ids = (eval_done_envs > 0).nonzero(as_tuple=False).squeeze(-1)
                        step_actions[done_env_ids, :] = 0

                        if "tactile" not in self.vec_env.observation_info:
                            # get tactile
                            current_tactile = self.vec_env.contact_forces.reshape(self.vec_env.num_envs, -1)
                            full_obs = torch.cat(
                                [
                                    (current_obs.reshape(1, self.vec_env.num_envs, -1)),
                                    current_tactile.reshape(1, self.vec_env.num_envs, -1),
                                ],
                                -1,
                            )
                        else:
                            full_obs = current_obs.reshape(1, self.vec_env.num_envs, -1).clone()
                        # get pcl obs
                        if self.vec_env.pcl_obs:
                            current_pcl_wrt_palm_obs = self.vec_env.object_pointclouds.reshape(
                                self.vec_env.num_envs, -1
                            )
                            # get full obs
                            full_obs = torch.cat(
                                [
                                    full_obs,
                                    current_pcl_wrt_palm_obs.reshape(1, self.vec_env.num_envs, -1),
                                ],
                                -1,
                            )
                        elif self.vec_env.real_pcl_obs:
                            current_real_pcl_obs = self.vec_env.rendered_pointclouds.reshape(self.vec_env.num_envs, -1)
                            # get full obs
                            full_obs = torch.cat(
                                [
                                    full_obs,
                                    current_real_pcl_obs.reshape(1, self.vec_env.num_envs, -1),
                                ],
                                -1,
                            )

                        if self.vec_env.img_pcl_obs:
                            current_img_pcl_obs = self.vec_env.imagined_pointclouds.reshape(self.vec_env.num_envs, -1)
                            # get full obs
                            full_obs = torch.cat(
                                [
                                    full_obs,
                                    current_img_pcl_obs.reshape(1, self.vec_env.num_envs, -1),
                                ],
                                -1,
                            )

                        self.demo_obs = torch.cat([self.demo_obs, full_obs.cpu()])

                        if "gf" in self.vec_env.observation_info:
                            self.vec_env.action_gf = grad.clone()
                        next_obs, rews, dones, infos = self.vec_env.step(step_actions)

                        self.demo_action = torch.cat(
                            [
                                self.demo_action,
                                self.vec_env.clamped_actions.clone().reshape(1, self.vec_env.num_envs, -1).cpu(),
                            ]
                        )
                        # collect abs action
                        # abs_arm_tran_action = self.vec_env.target_eef_pos.clone()
                        # abs_arm_rot_action = self.vec_env.target_eef_euler.clone()
                        # abs_hand_action = self.vec_env.curr_targets.clone()[:, self.vec_env.shadow_actuated_dof_indices]
                        # abs_action = torch.cat([abs_arm_tran_action, abs_arm_rot_action, abs_hand_action], -1)
                        # self.demo_abs_action = torch.cat(
                        #     [self.demo_abs_action, abs_action.reshape(1, self.vec_env.num_envs, -1)]
                        # )
                        # collect obs action
                        # obs_arm_tran_action = self.vec_env.rh_forearm_states[:, 0, :3].clone()
                        # obs_arm_quat_action = self.vec_env.rh_forearm_states[:, 0, 3:7].clone()
                        # obs_arm_rot_action = torch.stack(get_euler_xyz(obs_arm_quat_action), dim=1)
                        # obs_hand_action = self.vec_env.shadow_hand_dof_positions.clone()[
                        #     :, self.vec_env.shadow_actuated_dof_indices
                        # ]
                        # obs_action = torch.cat([obs_arm_tran_action, obs_arm_rot_action, obs_hand_action], -1)
                        # self.demo_obs_action = torch.cat(
                        #     [self.demo_obs_action, obs_action.reshape(1, self.vec_env.num_envs, -1)]
                        # )
                        current_obs.copy_(next_obs["obs"])

                        # done
                        new_done_env_ids = (dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                        if len(new_done_env_ids) > 0:
                            test_times += len(new_done_env_ids)
                            eval_done_envs[new_done_env_ids] = 1
                            success_env_ids = (
                                (rews >= self.vec_env.reach_goal_bonus).nonzero(as_tuple=False).squeeze(-1)
                            )
                            for success_env_id in success_env_ids:
                                if success_env_id in new_done_env_ids and self.cur_demo_num < self.total_demo_num:
                                    cur_demo = {}
                                    cur_demo["env_mode"] = self.vec_env.env_mode
                                    cur_demo["obs_space"] = self.vec_env.cfg["env"]["observationSpace"]
                                    cur_demo["action_space"] = self.vec_env.cfg["env"]["actionSpace"]
                                    cur_demo["ft_idx_in_all"] = self.vec_env.ft_idx_in_all
                                    if self.vec_env.pcl_obs:
                                        cur_demo["pcl_type"] = "full_pcl"
                                    elif self.vec_env.real_pcl_obs:
                                        cur_demo["pcl_type"] = "real_pcl"
                                    if self.vec_env.ur_control_type == "osc":
                                        arm_action_lower_limit = torch.ones(6, device=self.device) * -1.0
                                        arm_action_lower_limit[3:] *= torch.pi * 2
                                        hand_action_lower_limit = self.vec_env.gym_assets["current"]["robot"]["limits"][
                                            "lower"
                                        ][self.vec_env.shadow_actuated_dof_indices]
                                        robot_action_lower_limit = torch.cat(
                                            [arm_action_lower_limit, hand_action_lower_limit], -1
                                        )

                                        arm_action_upper_limit = torch.ones(6, device=self.device) * 1.0
                                        arm_action_upper_limit[3:] *= torch.pi * 2
                                        hand_action_upper_limit = self.vec_env.gym_assets["current"]["robot"]["limits"][
                                            "upper"
                                        ][self.vec_env.shadow_actuated_dof_indices]
                                        robot_action_upper_limit = torch.cat(
                                            [arm_action_upper_limit, hand_action_upper_limit], -1
                                        )

                                        cur_demo["dof_lower_limit"] = robot_action_lower_limit.cpu().numpy()
                                        cur_demo["dof_upper_limit"] = robot_action_upper_limit.cpu().numpy()

                                        # cur_demo["dof_lower_limit"] = (
                                        #     self.vec_env.gym_assets["current"]["robot"]["limits"]["lower"].cpu().numpy()
                                        # )
                                        # cur_demo["dof_upper_limit"] = (
                                        #     self.vec_env.gym_assets["current"]["robot"]["limits"]["upper"].cpu().numpy()
                                        # )
                                    cur_demo["init_obj_pos"] = self.demo_init_state["obj_pos"][success_env_id]
                                    cur_demo["init_obj_orn"] = self.demo_init_state["obj_orn"][success_env_id]
                                    cur_demo["init_robot_dof"] = self.demo_init_state["robot_dof"][success_env_id]
                                    cur_demo["target_dof"] = self.demo_init_state["target_dof"][success_env_id]
                                    cur_demo["obs"] = self.demo_obs[:, success_env_id, :].cpu().numpy()
                                    cur_demo["action"] = self.demo_action[:, success_env_id, :].detach().cpu().numpy()
                                    # cur_demo["abs_action"] = (
                                    #     self.demo_abs_action[:, success_env_id, :].detach().cpu().numpy()
                                    # )
                                    # cur_demo["obs_action"] = (
                                    #     self.demo_obs_action[:, success_env_id, :].detach().cpu().numpy()
                                    # )
                                    # get cur object code
                                    cur_demo_object_code = self.vec_env.occupied_object_codes[success_env_id]
                                    cur_demo["object_code"] = cur_demo_object_code
                                    cur_demo_object_grasp = self.vec_env.occupied_object_grasps[success_env_id]
                                    cur_demo["object_grasp"] = cur_demo_object_grasp

                                    if cur_demo_object_grasp in demo_per_grasp:
                                        demo_per_grasp[cur_demo_object_grasp] += 1
                                    else:
                                        demo_per_grasp[cur_demo_object_grasp] = 1

                                    if demo_per_grasp[cur_demo_object_grasp] <= self.args.collect_demo_num:
                                        # save demo using npy
                                        np.save(
                                            os.path.join(
                                                self.demo_dir,
                                                f"demo_{cur_demo_object_grasp}_{demo_per_grasp[cur_demo_object_grasp]}",
                                            ),
                                            cur_demo,
                                        )
                                        self.cur_demo_num += 1

                                    if self.total_demo_num < self.args.collect_demo_num * len(demo_per_grasp):
                                        print("total demo num is less than collect demo num")
                                        print("total demo num: ", self.total_demo_num)
                                        print("expected demo num: ", self.args.collect_demo_num * len(demo_per_grasp))
                                        self.total_demo_num = self.args.collect_demo_num * len(demo_per_grasp)
                                    # if cur_demo_object_code in demo_per_obj:
                                    #     demo_per_obj[cur_demo_object_code] += 1
                                    # else:
                                    #     demo_per_obj[cur_demo_object_code] = 1

                                    # if demo_per_obj[cur_demo_object_code] <= self.args.collect_demo_num:
                                    #     # save demo using npy
                                    #     np.save(
                                    #         os.path.join(
                                    #             self.demo_dir,
                                    #             f"demo_{cur_demo_object_code}_{demo_per_obj[cur_demo_object_code]}",
                                    #         ),
                                    #         cur_demo,
                                    #     )
                                    #     self.cur_demo_num += 1

                        if eval_done_envs.sum() == self.vec_env.num_envs or self.cur_demo_num == self.total_demo_num:
                            break
        else:
            # start evaluation
            with tqdm(total=self.eval_round) as pbar:
                pbar.set_description("Validating:")
                with torch.no_grad():
                    for r in range(self.eval_round):
                        if save_video and r <= save_time:
                            all_images = torch.tensor([], device=self.device)
                        # reset env
                        current_obs = self.vec_env.reset()["obs"]
                        eval_done_envs = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)
                        successes = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
                        eps_len = torch.zeros(self.vec_env.num_envs, dtype=torch.long, device=self.device)
                        pos_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        rot_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        contact_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        succ_eps_len = []
                        if plot_direction:
                            arm_diff_direction = []
                            hand_diff_direction = []

                        #     contact_dist = torch.zeros(self.vec_env.num_envs, device=self.device)
                        tran_rew = 0
                        rot_rew = 0
                        contact_rew = 0
                        height_rew = 0

                        if save_state:
                            self.eval_metrics["gf_state_init"].append(self.vec_env.get_states(gf_state=True))
                            self.eval_metrics["gf_state_gt"].append(self.vec_env.target_hand_dof)

                        # step
                        while True:
                            # Compute the action
                            actions, grad, _ = self.compute_action(current_obs=current_obs, mode="eval")
                            step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())
                            if self.vec_env.progress_buf[0] == 49 and save_state:
                                self.eval_metrics["gf_state_final"].append(self.vec_env.get_states(gf_state=True))

                            # Step the vec_environment
                            # print(step_actions)
                            # set done env action to be zero
                            done_env_ids = (eval_done_envs > 0).nonzero(as_tuple=False).squeeze(-1)
                            step_actions[done_env_ids, :] = 0

                            if "gf" in self.vec_env.observation_info:
                                self.vec_env.action_gf = grad.clone()
                            next_obs, rews, dones, infos = self.vec_env.step(step_actions)

                            if plot_direction:
                                arm_diff_direction.append(
                                    torch.mean(infos["arm_pos_diff_direction"] + infos["arm_rot_diff_direction"]).item()
                                    / 6
                                )
                                hand_diff_direction.append(torch.mean(infos["hand_diff_direction"]).item() / 20)

                            if save_video and r <= save_time:
                                image = self.vec_env.render(
                                    rgb=True, img_size=img_size, vis_env_num=self.vis_env_num
                                ).reshape(self.vis_env_num, 1, img_size, img_size, 3)
                                all_images = torch.cat([all_images, image], 1)
                            current_obs.copy_(next_obs["obs"])

                            # done
                            new_done_env_ids = (dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
                            if len(new_done_env_ids) > 0:
                                # if 0 in new_done_env_ids:
                                #     print("--")
                                if r > save_time and save_video:
                                    self.vec_env.graphics_device_id = -1
                                    self.vec_env.enable_camera_sensors = False

                                if save_video and r <= save_time:
                                    for i, images in enumerate(all_images):
                                        obj_type = self.vec_env.object_type_per_env[i]
                                        save_path = os.path.join(
                                            self.video_log_dir, f"{obj_type}_epoach:{it}_round:{r}"
                                        )
                                        images_to_video(
                                            path=save_path, images=images.cpu().numpy(), size=(img_size, img_size)
                                        )

                                test_times += len(new_done_env_ids)
                                reward_all.extend(rews[new_done_env_ids].cpu().numpy())
                                pos_dist[new_done_env_ids] = infos["pos_dist"][new_done_env_ids]
                                rot_dist[new_done_env_ids] = infos["rot_dist"][new_done_env_ids]
                                contact_dist[new_done_env_ids] = infos["fj_dist"][new_done_env_ids]

                                eval_done_envs[new_done_env_ids] = 1
                                success_env_ids = (
                                    (rews >= self.vec_env.reach_goal_bonus).nonzero(as_tuple=False).squeeze(-1)
                                )
                                successes[success_env_ids] = 1
                                eps_len[new_done_env_ids] = self.vec_env.progress_buf[new_done_env_ids]
                                succ_eps_len.extend(self.vec_env.progress_buf[success_env_ids].cpu().numpy().tolist())

                            if test_times == (r + 1) * self.vec_env.num_envs:
                                # self.vec_env.lift_test(eval_done_envs.nonzero(as_tuple=False).squeeze(-1))
                                # for id in self.vec_env.successes.nonzero(as_tuple=False):
                                #     print(fj_dist[id], contact_dist[id], pos_dist[id], rot_dist[id])
                                assert torch.sum(eval_done_envs).item() == self.vec_env.num_envs
                                success_rates.append(infos["success_num"] / self.vec_env.num_envs)
                                eps_len_all.append(eps_len.float().mean().item())
                                succ_eps_len_all.append(np.mean(succ_eps_len))

                                if plot_direction:
                                    plt.plot(arm_diff_direction, label="arm_diff_direction")
                                    plt.plot(hand_diff_direction, label="hand_diff_direction")

                                if save_metric:
                                    self.eval_metrics["time_step"].append(it)
                                    self.eval_metrics["success_rate"].append(
                                        float((infos["success_num"] / self.vec_env.num_envs).cpu().numpy())
                                    )
                                    if self.vec_env.env_mode == "relpose":
                                        self.eval_metrics["pos_dist"].append(torch.mean(pos_dist).item())
                                        self.eval_metrics["rot_dist"].append(torch.mean(rot_dist).item())
                                    for occupied_object_id, occupied_object_code in enumerate(
                                        self.vec_env.occupied_object_codes
                                    ):
                                        obj_id_all = self.eval_metrics["obj_shapes"].index(occupied_object_code)
                                        self.eval_metrics["success_nums"][obj_id_all] += self.vec_env.successes[
                                            occupied_object_id
                                        ]
                                        self.eval_metrics["num_trials"][obj_id_all] += 1

                                    for i, (code, grasp) in enumerate(
                                        zip(self.vec_env.occupied_object_codes, self.vec_env.occupied_object_grasps)
                                    ):
                                        if (code, grasp) in self.eval_metrics["grasp"]:
                                            idx = self.eval_metrics["grasp"].index((code, grasp))
                                        else:
                                            idx = len(self.eval_metrics["grasp"])
                                            self.eval_metrics["grasp"].append((code, grasp))
                                            self.eval_metrics["num_trails_per_grasp"].append(0)
                                            self.eval_metrics["num_success_per_grasp"].append(0)
                                        self.eval_metrics["num_trails_per_grasp"][idx] += 1
                                        self.eval_metrics["num_success_per_grasp"][idx] += self.vec_env.successes[i]

                                    # self.eval_metrics['stability'].append(float(infos['stability'].cpu().numpy()))
                                    if self.vec_env.mode == "eval":
                                        with open(
                                            f"logs/{self.args.exp_name}/metrics_{self.args.eval_name}_eval_{self.args.seed}.pkl",
                                            "wb",
                                        ) as f:
                                            pickle.dump(self.eval_metrics, f)
                                    else:
                                        with open(
                                            os.path.join(
                                                self.log_dir, f"metric_{self.args.exp_name}_{self.args.seed}.pkl"
                                            ),
                                            "wb",
                                        ) as f:
                                            pickle.dump(self.eval_metrics, f)

                                break
                        pbar.update(1)

            assert test_times == self.eval_round * self.vec_env.num_envs
            # plt.bar(range(len(self.eval_metrics["obj_shapes"])), self.eval_metrics["success_nums"])
            success_rates = torch.cat(success_rates)
            sr_mu, sr_std = success_rates.mean().cpu().numpy().item(), success_rates.std().cpu().numpy().item()
            print(f"|| num_envs: {self.vec_env.num_envs} || eval_times: {self.eval_round}")
            print(f"eval_success_rate % : {sr_mu*100:.2f} +- {sr_std*100:.2f}")
            eval_rews = np.mean(reward_all)
            print(f"eval_rewards: {eval_rews}")
            print(f"eval_eps_len: {np.mean(eps_len_all)}")
            print(f"eval_succ_eps_len: {np.mean(succ_eps_len_all)}")
            self.writer.add_scalar("Eval/success_rate", sr_mu, it)
            self.writer.add_scalar("Eval/eval_rews", eval_rews, it)

    def run(self, num_learning_iterations, log_interval=1):
        if self.is_testing:
            self.eval(0)
        else:
            # train initilization
            self.actor_critic.train()
            self.vec_env.train()
            rewbuffer = deque(maxlen=100)
            lenbuffer = deque(maxlen=100)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []

            # reset env
            current_obs = self.vec_env.reset()["obs"]
            current_states = self.vec_env.get_state()
            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()["obs"]
                        current_states = self.vec_env.get_state()

                    # Compute the action
                    actions, actions_log_prob, values, mu, sigma, grad, storage_obs = self.compute_action(
                        current_obs=current_obs, current_states=current_states
                    )
                    step_actions = self.process_actions(actions=actions.clone(), grad=grad.clone())

                    # Step the vec_environment
                    if "gf" in self.vec_env.observation_info:
                        self.vec_env.action_gf = grad.clone()
                    next_obs, rews, dones, infos = self.vec_env.step(step_actions)
                    next_states = self.vec_env.get_state()

                    # Record the transition
                    self.storage.add_transitions(
                        storage_obs, current_states, actions, rews, dones, values, actions_log_prob, mu, sigma
                    )
                    current_obs.copy_(next_obs["obs"])
                    current_states.copy_(next_states)

                    # Book keeping
                    ep_infos.append(infos.copy())
                    # set_trace()

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    # done
                    if torch.sum(dones) > 0:
                        current_obs = self.vec_env.reset(dones)["obs"]
                        current_states = self.vec_env.get_state()

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)

                _, _, last_values, _, _, _, _ = self.compute_action(
                    current_obs=current_obs, current_states=current_states, mode="train"
                )
                stop = time.time()
                collection_time = stop - start
                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update()
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if (it + 1) % log_interval == 0:
                    self.actor_critic.eval()
                    self.eval(it + 1)
                    self.actor_critic.train()
                    self.vec_env.train()
                    self.save(os.path.join(self.log_dir, "model_{}.pt".format(it + 1)))

                    current_obs = self.vec_env.reset()["obs"]
                    current_states = self.vec_env.get_state()
                    cur_episode_length[:] = 0
                    # TODO clean extras
                ep_infos.clear()
            self.save(os.path.join(self.log_dir, "model_{}.pt".format(num_learning_iterations)))

    def log(self, locs, width=70, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if key == "success_num":
                    value = torch.sum(infotensor)
                    self.writer.add_scalar("Episode/" + "total_success_num", value, locs["it"])
                    ep_string += f"""{f'Total episode {key}:':>{pad}} {value:.4f}\n"""
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        self.writer.add_scalar("Train2/mean_reward/step", locs["mean_reward"], locs["it"])
        self.writer.add_scalar("Train2/mean_episode_length/episode", locs["mean_trajectory_length"], locs["it"])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)

        if self.args.exp_name == "ilad":
            iteration_count = 0

        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):

            for indices in batch:
                # print(len(indices))
                if self.args.exp_name == "ilad":
                    iteration_count += 1

                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                if self.args.exp_name == "ilad":
                    """Get demo advantages."""
                    pass
                    # demo_obs_batch = self.demo_storage.observations.view(-1, self.demo_storage.observations.size()[2:])[indices]
                    # demo_states_batch = self.demo_storage.states.view(-1, *self.demo_storage.size()[2:])[indices]
                    # demo_actions_batch = self.demo_storage.actions.view(-1, self.demo_storage.size(-1))[indices]
                    # demo_advantages_batch = self.demo_storage.advantages.view(-1, 1)[indices]
                    # demo_old_actions_log_prob_batch = self.demo_storage.actions_log_prob.view(-1, 1)[indices]
                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(
                    obs_batch, states_batch, actions_batch
                )

                if self.args.exp_name == "ilad":
                    """Evaulate demo."""
                    n = len(indices)
                    # demo_indices = torch.randint(0, self.demo_storage.observations.size(0), (n,))
                    demo_indices = torch.randint(0, self.dataset.num_samples, (n,))
                    demo_observations, demo_actions = self.dataset.get_random_tuple(demo_indices)

                    demo_observations = demo_observations.to(self.device)
                    demo_actions = demo_actions.to(self.device)

                    demo_log_likelihood = self.actor_critic.cal_actions_log_prob(demo_observations, demo_actions)[1]
                    demo_advantages = self.compute_demo_advantages(demo_observations, demo_actions)

                    demo_weights = (demo_log_likelihood - torch.min(demo_log_likelihood)) / (
                        torch.max(demo_log_likelihood) - torch.min(demo_log_likelihood)
                    )
                    demo_advantages = (
                        self.lambda0 * (self.lambda1**iteration_count) * demo_weights
                        + self.lambda2 * (1 - (self.lambda3**iteration_count)) * demo_advantages
                    )

                # KL
                if self.desired_kl != None and self.schedule == "adaptive":
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.args.exp_name == "ilad":
                    # demo_ratio = torch.exp(demo_actions_log_prob_batch - torch.squeeze(demo_old_actions_log_prob_batch))
                    # demo_surrogate = -torch.squeeze(demo_advantages_batch) * demo_ratio
                    # demo_surrogate_clipped = -torch.squeeze(demo_advantages_batch) * torch.clamp(demo_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    # demo_surrogate_loss = torch.max(demo_surrogate, demo_surrogate_clipped).mean()
                    demo_surrogate_loss = -torch.mean(demo_advantages)
                    # self.optimizer.zero_grad()
                    # demo_surrogate_loss.backward()
                    # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    # self.optimizer.step()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                if self.args.exp_name == "ilad":
                    """Add demo_surrogate_loss."""
                    loss = loss + demo_surrogate_loss
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.args.exp_name == "ilad":
                    self.fit_demo_advantage_func(obs_batch, actions_batch, advantages_batch)

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    """
    ILAD
    """

    def compute_demo_advantages(self, demo_observations, demo_actions):
        baseline = self.actor_critic.forward_critic(demo_observations)
        estimated_value = self.actor_critic.forward_additional_critic(demo_observations, demo_actions)
        advantages = estimated_value - baseline
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages + 1.0
        return advantages

    def fit_demo_advantage_func(self, observations, actions, advantages):
        estimated_advantages = self.actor_critic.forward_additional_critic(observations, actions)
        loss = (estimated_advantages - advantages).pow(2).mean()
        self.additional_critic_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.additional_critic_optimizer.step()

    """
    utils
    """

    def grad_norm(self, grad):
        if len(grad.shape) == 2:
            scale_grad = (torch.max((abs(grad)), dim=1)[0]).reshape(-1, 1).expand_as(grad)
        elif len(grad.shape) == 3:
            scale_grad = (torch.max((abs(grad)), dim=2)[0]).reshape(grad.shape[0], grad.shape[1], 1).expand_as(grad)
        grad = grad / scale_grad
        return grad

    # TODO action type
    def action2grad(self, x, action_type, inv=False, relative=True, cur_x=None):
        if not inv:
            batch_size = x.size(0)
            state_dim = x.size(1)
            x = torch.cat(
                [torch.sin(x).reshape(batch_size, state_dim, 1), torch.cos(x).reshape(batch_size, state_dim, 1)], 2
            ).reshape(batch_size, -1)
            return x
        else:
            batch_size = x.size(0)
            state_dim = x.size(1)
            x = x.reshape(batch_size, int(state_dim / 2), 2)
            cur_x = cur_x.reshape(batch_size, int(state_dim / 2), 2)

            cur_x = torch.cat([-cur_x[:, :, 0:1], cur_x[:, :, 1:2]], dim=-1)
            ori_grad = torch.sum(torch.cat([x[:, :, 1:2], x[:, :, 0:1]], dim=-1) * cur_x, dim=-1, keepdim=True).reshape(
                batch_size, int(state_dim / 2)
            )
            return ori_grad

    def get_obs_with_grad(self, current_obs):
        current_obs = current_obs.reshape(self.vec_env.num_envs, self.args.n_obs_steps, -1)
        # compute score
        B = current_obs.size(0)
        current_state = current_obs[:, :, :obs_state_dim].clone().to(self.device).float()
        current_obj_pcl_2h = (
            current_obs[:, :, obs_state_dim : obs_state_dim + 3 * pcl_number]
            .clone()
            .to(self.device)
            .float()
            .reshape(self.vec_env.num_envs, self.args.n_obs_steps, pcl_number, 3)
        )
        current_obj_pcl_2h = current_obj_pcl_2h.permute(0, 1, 3, 2)
        start = self.args.n_obs_steps - 1
        end = start + self.args.n_action_steps

        batch_time_step = torch.ones(B, device=self.device).unsqueeze(1) * self.t0

        if self.args.space == "riemann":
            score_action = action2grad(self.current_action, action_type=self.args.score_action_type)
            grad = torch.zeros(B, self.args.n_prediction_steps, self.vec_env.num_actions * 2, device=self.device)
        else:
            score_action = self.current_action.clone()
            grad = torch.zeros(B, self.args.n_prediction_steps, self.vec_env.num_actions, device=self.device)

        bz = 256
        iter_num = int(np.ceil(B / bz))
        for order in range(iter_num):
            with torch.no_grad():
                tmp_grad = self.score(
                    (
                        (
                            score_action[order * bz : (order + 1) * bz, :],
                            current_obj_pcl_2h[order * bz : (order + 1) * bz, :],
                            current_state[order * bz : (order + 1) * bz, :],
                        )
                    ),
                    batch_time_step[order * bz : (order + 1) * bz, :].unsqueeze(1),
                ).detach()
                if self.args.space == "riemann":
                    grad[order * bz : (order + 1) * bz, :] = self.action2grad(
                        tmp_grad, inv=True, action_type=self.args.score_action_type
                    ).clone()
                else:
                    grad[order * bz : (order + 1) * bz, :] = tmp_grad.clone()

        if self.args.action_mode == "abs" or self.args.action_mode == "obs":
            raise NotImplementedError

        if self.grad_process is not None:
            if "norm" in self.grad_process:
                grad = self.grad_norm(grad)

        step_actions = grad[:, start:end, :]
        self.current_action[:, 0, :] = step_actions[:, -1, :].clone()
        # TODO only last frame for rl, only one step for grad
        current_obs[:, -1:, -self.vec_env.num_actions :] = step_actions.clone()

        if self.args.n_action_steps == 1:
            return current_obs[:, -1, :], step_actions[:, -1, :].clone()

    def process_actions(self, actions, grad=None):
        if self.action_type == "direct":
            step_actions = actions
        elif self.action_type == "gf":
            step_actions = grad
        elif self.action_type == "joint":
            if self.sub_action_type == "add+jointscale":
                self.vec_env.extras["grad_ss_mean"] = torch.mean(abs(actions[:, : self.vec_env.num_actions]), -1)
                self.vec_env.extras["grad_ss_std"] = torch.std(abs(actions[:, : self.vec_env.num_actions]), -1)
                self.vec_env.extras["residual_mean"] = torch.mean(abs(actions[:, self.vec_env.num_actions :]), -1)
                self.vec_env.extras["residual_std"] = torch.std(abs(actions[:, self.vec_env.num_actions :]), -1)
                step_actions = grad * actions[:, : self.vec_env.num_actions] + actions[:, self.vec_env.num_actions :]
            elif self.sub_action_type == "addscale+add":
                step_actions = (
                    grad * (actions[:, :1] + actions[:, 1 : 1 + self.vec_env.num_actions])
                    + actions[:, 1 + self.vec_env.num_actions :]
                )
        return step_actions

    def compute_action(self, current_obs, current_states=None, mode="train"):
        if "gf" in self.vec_env.observation_info:
            current_obs, grad = self.get_obs_with_grad(current_obs)
        else:
            grad = torch.tensor([], device=self.device)

        if self.actor_critic.pcl_dim > 0 and self.pointnet_finetune:
            batch_num = current_obs.size(0) // self.finetune_pointnet_bz + 1
            for _ in range(batch_num):
                current_obs_batch = current_obs[self.finetune_pointnet_bz * _ : self.finetune_pointnet_bz * (_ + 1), :]
                # current_states_batch = current_states[:,self.finetune_pointnet_bz*batch_num+self.finetune_pointnet_bz*(batch_num+1)]
                if mode == "train":
                    actions_batch, actions_log_prob_batch, values_batch, mu_batch, sigma_batch = self.actor_critic.act(
                        current_obs_batch, current_states
                    )
                else:
                    actions_batch = self.actor_critic.act_inference(current_obs_batch)

                if _ == 0:
                    if mode == "train":
                        actions, actions_log_prob, values, mu, sigma = (
                            actions_batch,
                            actions_log_prob_batch,
                            values_batch,
                            mu_batch,
                            sigma_batch,
                        )
                    else:
                        actions = actions_batch
                else:
                    if mode == "train":
                        actions = torch.cat([actions, actions_batch])
                        actions_log_prob = torch.cat([actions_log_prob, actions_log_prob_batch])
                        values = torch.cat([values, values_batch])
                        mu = torch.cat([mu, mu_batch])
                        sigma = torch.cat([sigma, sigma_batch])
                    else:
                        actions = torch.cat([actions, actions_batch])
        else:
            if mode == "train":
                actions, actions_log_prob, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
            else:
                actions = self.actor_critic.act_inference(current_obs)

        if mode == "train":
            return actions, actions_log_prob, values, mu, sigma, grad, current_obs
        else:
            return actions, grad, current_obs
