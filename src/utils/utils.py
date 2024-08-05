import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset

from algorithms.SDE import action2grad, cond_ode_sampler, pc_sampler
from tasks.torch_utils import get_euler_xyz

real_pcl_number = 512
imagined_pcl_number = 512
pcl_number = real_pcl_number + imagined_pcl_number


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True


def save_video(imgs, save_path, simulation=False, fps=50, render_size=256, suffix="mp4"):
    # states: [state, ....]
    # state: (60, )
    if suffix == "gif":
        from PIL import Image

        images_to_gif(
            save_path + f".{suffix}", [Image.fromarray(img[:, :, ::-1], mode="RGB") for img in imgs], fps=len(imgs) // 5
        )
    else:
        batch_imgs = np.stack(imgs, axis=0)
        images_to_video(save_path + f".{suffix}", batch_imgs, fps, (render_size, render_size))


def images_to_gif(path, images, fps):
    images[0].save(path, save_all=True, append_images=images[1:], fps=fps, loop=0)


def images_to_video(path, images, fps, size):
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size, isColor=True)
    for item in images:
        out.write(item.astype(np.uint8))
    out.release()


def get_dict_key(dic, value):
    key = list(dic.keys())[list(dic.values()).index(value)]
    return key


# torch utils
def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)


def normalize(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Same as `scale_transform` but with a different name."""
    return scale_transform(x, lower, upper)


def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset


def denormalize(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Same as `unscale_transform` but with a different name."""
    return unscale_transform(x, lower, upper)


def grad_norm(grad):
    if len(grad.shape) == 2:
        scale_grad = (torch.max((abs(grad)), dim=1)[0]).reshape(-1, 1).expand_as(grad)
    elif len(grad.shape) == 3:
        scale_grad = (torch.max((abs(grad)), dim=2)[0]).reshape(grad.shape[0], grad.shape[1], 1).expand_as(grad)
    grad = grad / scale_grad
    return grad


def eval_policy(
    init_robot_dof,
    init_obj_pos,
    init_obj_orn,
    object_targets,
    env,
    dataset,
    score,
    epoch,
    r,
    test_times,
    device,
    video_path,
    expert=None,
    eval_mode=None,
    action_steps=1,
    sampler=None,
    random_action=False,
    visualizer=None,
    args=None,
    prior_fn=None,
    sde_fn=None,
    score_mode="SDE",
):
    imgs = []
    eval_done_envs = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    successes = torch.zeros(env.num_envs, dtype=torch.float, device=device)
    pos_dist = torch.zeros(env.num_envs, device=device)
    rot_dist = torch.zeros(env.num_envs, device=device)
    contact_dist = torch.zeros(env.num_envs, device=device)
    min_pos_dist = torch.ones(env.num_envs, device=device)
    min_rot_dist = torch.ones(env.num_envs, device=device)
    min_contact_dist = torch.ones(env.num_envs, device=device)
    high_similarities = torch.zeros(env.num_envs, device=device)
    current_action = torch.randn(env.num_envs, dataset.horizon, dataset.action_dim).to(device).float()
    eps_len = torch.zeros(env.num_envs, device=device)

    current_obs = env.reset()["obs"].reshape(env.num_envs, dataset.n_obs_steps, -1).clone()
    # set init state
    env.set_states(robot_dof=init_robot_dof, object_targets=object_targets, obj_pos=init_obj_pos, obj_orn=init_obj_orn)

    # step with zero action to get current obs
    next_obs, rews, dones, infos = env.step(torch.zeros(env.num_envs, dataset.full_action_dim).to(device).float())
    current_obs.copy_(next_obs["obs"].reshape(env.num_envs, dataset.n_obs_steps, -1))

    assert current_obs.size(-1) == score.full_obs_dim
    # eval learned policy
    while True:
        if "policy" in eval_mode:
            # divide current obs
            current_state = current_obs[:, :, : dataset.obs_state_dim].clone().to(device).float()
            current_obj_pcl_2h = (
                current_obs[:, :, dataset.obs_state_dim : dataset.obs_state_dim + dataset.obs_pcl_dim]
                .clone()
                .to(device)
                .float()
                .reshape(env.num_envs, dataset.n_obs_steps, pcl_number, 3)
            )
            current_obj_pcl_2h = current_obj_pcl_2h.permute(0, 1, 3, 2)

            if random_action:
                current_action = torch.randn(env.num_envs, dataset.horizon, dataset.action_dim).to(device).float()
            elif args.action_mode == "abs":
                abs_arm_tran_action = env.target_eef_pos.clone()
                abs_arm_rot_action = env.target_eef_euler.clone()
                abs_hand_action = env.curr_targets.clone()[:, env.hand_actuated_dof_indices]
                abs_action = torch.cat([abs_arm_tran_action, abs_arm_rot_action, abs_hand_action], -1)
                current_action = abs_action.unsqueeze(1).repeat(1, dataset.horizon, 1)
            elif args.action_mode == "obs":
                obs_arm_tran_action = env.endeffector_positions.clone()
                obs_arm_quat_action = env.endeffector_orientations.clone()
                obs_arm_rot_action = torch.stack(get_euler_xyz(obs_arm_quat_action), dim=1)
                obs_hand_action = env.shadow_hand_dof_positions.clone()[:, env.hand_actuated_dof_indices]
                obs_action = torch.cat([obs_arm_tran_action, obs_arm_rot_action, obs_hand_action], -1)
                current_action = obs_action.unsqueeze(1).repeat(1, dataset.horizon, 1)

            if args.cond_on_arm and args.action_type == "hand":
                arm_action = torch.tensor([]).to(device).float()
                for obs_step in range(dataset.n_obs_steps):
                    e_action = torch.clamp(
                        expert.get_action(current_obs[:, obs_step, : dataset.obs_state_dim], mode="eval")[
                            :, :6
                        ].reshape(1, env.num_envs, -1),
                        -env.clip_actions,
                        env.clip_actions,
                    )
                    arm_action = torch.cat([arm_action, e_action])
                current_state = torch.cat([current_state, arm_action.transpose(0, 1)], dim=-1)
            # get step actions
            # o2 -> a2, thus n_onb_steps - 1
            start = dataset.n_obs_steps - 1
            end = start + action_steps

            if score_mode == "SDE":
                if sampler == "ode":
                    in_process_sample, res = cond_ode_sampler(
                        score,
                        prior_fn,
                        sde_fn,
                        (current_action, current_obj_pcl_2h, current_state),
                        t0=args.t0,
                        device=device,
                        num_steps=500,
                        is_random=False,
                        batch_size=env.num_envs,
                        hand_pcl=args.hand_pcl,
                        full_state=None,
                        envs=env,
                        hand_model=None,
                        relative=args.relative,
                        action_type=args.action_type,
                        space=args.space,
                    )
                    step_actions = res[:, start:end, :].to(torch.float32)
                elif sampler == "pc":
                    res = pc_sampler(
                        score,
                        prior_fn,
                        sde_fn,
                        (current_action, current_obj_pcl_2h, current_state),
                        t0=args.t0,
                        device=device,
                        num_steps=500,
                        is_random=False,
                        relative=args.relative,
                        action_type=args.action_type,
                        space=args.space,
                    )
                    step_actions = res[:, start:end, :].to(torch.float32)
                else:
                    batch_time_step = torch.ones(env.num_envs, device=device).unsqueeze(1) * args.t0
                    if args.space == "riemann":
                        score_action = action2grad(current_action, action_type=args.action_type)
                    else:
                        score_action = current_action.clone()
                    if args.dual_score:
                        # magic number
                        arm_grad = score["arm"](
                            (score_action[:, :, :6], current_obj_pcl_2h, current_state), batch_time_step.unsqueeze(1)
                        )
                        if args.cond_on_arm:
                            arm_action = grad_norm(arm_grad.clone())
                            current_state = torch.cat([current_state, arm_action[:, : dataset.n_obs_steps, :]], dim=-1)
                        hand_grad = score["hand"](
                            (score_action[:, :, 6:26], current_obj_pcl_2h, current_state), batch_time_step.unsqueeze(1)
                        )
                        grad = torch.cat([arm_grad, hand_grad], dim=-1)
                    else:
                        grad = score((score_action, current_obj_pcl_2h, current_state), batch_time_step.unsqueeze(1))
                    if args.space == "riemann":
                        grad = action2grad(grad, inv=True, action_type=args.action_type)
                    if args.action_mode == "abs" or args.action_mode == "obs":
                        if args.action_type == "arm":
                            grad = (
                                grad * torch.tensor(dataset.action_range[:6]).to(device).float() * 0.5
                            )  # magic number
                        elif args.action_type == "hand":
                            grad = (
                                grad * torch.tensor(dataset.action_range[6:26]).to(device).float() * 0.5
                            )  # magic number
                        elif args.action_type == "all":
                            grad = grad * torch.tensor(dataset.action_range).to(device).float() * 0.5
                    # print(torch.mean(abs(grad)))
                    if "norm" in args.grad_process or "norm" in eval_mode:
                        grad = grad_norm(grad)

                    step_actions = grad[:, start:end, :]
            elif score_mode == "DDIM":
                res = score.predict_action((current_action, current_obj_pcl_2h, current_state), random_traj=False)
                step_actions = res[:, start:end, :].to(torch.float32)
            if not random_action:
                if args.action_mode == "rel":
                    current_action = torch.zeros(env.num_envs, dataset.horizon, dataset.action_dim).to(device).float()
                    # current_action[:, :dataset.n_obs_steps, :] = grad[:, end-1:end-1+dataset.n_obs_steps, :].clone()
                    current_action[:, :, :] = step_actions[:, -1:, :].clone()

        # step actions simulation
        for sub_action in range(action_steps):
            # get expert actions
            expert_step_actions = expert.get_action(current_obs[:, -1, : dataset.obs_state_dim], mode="eval")
            if "policy" in eval_mode:
                # policy actions
                policy_actions = step_actions[:, sub_action, :]
                # subsitude expert actions with policy actions according to action type
                if args.action_type == "arm":
                    expert_step_actions[:, :6] = policy_actions
                elif args.action_type == "hand":
                    # get the num of different direction of policy_actions and expert_step_actions
                    # diff_direction_num = torch.sum(policy_actions*expert_step_actions[:, 6:26]<0, dim=1)
                    # print(diff_direction_num)
                    expert_step_actions[:, 6:26] = policy_actions
                elif args.action_type == "all":
                    expert_step_actions = policy_actions

            done_env_ids = (eval_done_envs > 0).nonzero(as_tuple=False).squeeze(-1)
            expert_step_actions[done_env_ids, :] = 0

            if (args.sampler == "ode" or args.sampler == "pc") and (
                args.action_mode == "abs" or args.action_mode == "obs"
            ):
                env.set_states(robot_dof=expert_step_actions, step_time=-1, set_dof_state=False, arm_ik=True)
                next_obs, rews, dones, infos = env.step(expert_step_actions * 0)
            else:
                next_obs, rews, dones, infos = env.step(expert_step_actions)

            # target pos dist is 0.01, rot dist is 0.1, contact dist is 0.15
            # compute current pos dist, rot dist, contact dist how close to target
            pos_similarity = 0.01 / (infos["pos_dist"] + 0.01)
            rot_similarity = 0.1 / (infos["rot_dist"] + 0.1)
            contact_similarity = 0.15 / (infos["contact_dist"] + 0.15)
            all_similarity = pos_similarity * rot_similarity * contact_similarity
            # get env id which is higher than current high_similarities
            higher_env_ids = (all_similarity > high_similarities).nonzero(as_tuple=False).squeeze(-1)
            min_pos_dist[higher_env_ids] = infos["pos_dist"][higher_env_ids]
            min_rot_dist[higher_env_ids] = infos["rot_dist"][higher_env_ids]
            min_contact_dist[higher_env_ids] = infos["contact_dist"][higher_env_ids]
            high_similarities[higher_env_ids] = all_similarity[higher_env_ids]

            if args.web_visualizer_port != -1:
                # TODO enable for multiple env
                visualizer.set_cam_pose([0.8, 0, 1.2], [0, 0, 0.5])
                imgs.append(visualizer.render())

            new_done_env_ids = (dones & (1 - eval_done_envs)).nonzero(as_tuple=False).squeeze(-1)
            if len(new_done_env_ids) > 0:
                test_times += len(new_done_env_ids)
                eval_done_envs[new_done_env_ids] = 1
                success_env_ids = (rews >= env.reach_goal_bonus).nonzero(as_tuple=False).squeeze(-1)
                successes[success_env_ids] = 1
                pos_dist[new_done_env_ids] = infos["pos_dist"][new_done_env_ids]
                rot_dist[new_done_env_ids] = infos["rot_dist"][new_done_env_ids]
                contact_dist[new_done_env_ids] = infos["contact_dist"][new_done_env_ids]
                eps_len[new_done_env_ids] = env.progress_buf[new_done_env_ids].to(torch.float32)

            current_obs.copy_(next_obs["obs"].reshape(env.num_envs, dataset.n_obs_steps, -1))

        if test_times == (r + 1) * env.num_envs:
            break

    if args.web_visualizer_port != -1:
        save_video(
            imgs=imgs,
            save_path=os.path.join(video_path, f"epoch:{epoch}_round:{r}_{eval_mode}_sa:{action_steps}"),
            suffix="gif",
        )

    return (
        successes.mean().item(),
        min_pos_dist.mean().item(),
        min_rot_dist.mean().item(),
        min_contact_dist.mean().item(),
        eps_len.mean().item(),
        min_pos_dist.std().item(),
        min_rot_dist.std().item(),
        min_contact_dist.std().item(),
        eps_len.std().item(),
    )


def eval_state(
    eval_dataset,
    eval_dataloader,
    score,
    epoch,
    device,
    env,
    args,
    prior_fn,
    sde_fn,
    visualizer,
    video_path,
    real_obs=True,
):
    imgs = []
    demo_path = "data/expert_dataset/eval/demo_bottle_s118_0.npy"
    demo = np.load(demo_path, allow_pickle=True).tolist()
    init_obj_pos = torch.tensor(demo["init_obj_pos"], dtype=torch.float32, device=env.device).reshape(1, -1)
    init_obj_orn = torch.tensor(demo["init_obj_orn"], dtype=torch.float32, device=env.device).reshape(1, -1)
    init_robot_dof = torch.tensor(demo["init_robot_dof"], dtype=torch.float32, device=env.device).reshape(1, -1)
    object_targets = torch.tensor(demo["target_dof"], dtype=torch.float32, device=env.device).reshape(1, -1)

    obs = torch.tensor(demo["obs"], dtype=torch.float32, device=env.device)
    action = torch.tensor(demo["action"], dtype=torch.float32, device=env.device)
    abs_action = torch.tensor(demo["abs_action"], dtype=torch.float32, device=env.device)
    dof_lower_limits = torch.tensor(demo["dof_lower_limit"], dtype=torch.float32, device=env.device)
    dof_upper_limits = torch.tensor(demo["dof_upper_limit"], dtype=torch.float32, device=env.device)

    if real_obs:
        current_obs = env.reset()["obs"].reshape(env.num_envs, eval_dataset.n_obs_steps, -1).clone()

    env.set_states(robot_dof=init_robot_dof, object_targets=object_targets, obj_pos=init_obj_pos, obj_orn=init_obj_orn)

    # if real_obs:
    #     next_obs, rews, dones, infos = env.step(action[0:1] * 0)

    for i in range(len(obs) - 1):
        cur_obs = obs[i : i + args.n_obs_steps].reshape(1, args.n_obs_steps, -1)
        state = cur_obs[:, :, : eval_dataset.obs_state_dim].clone().to(device).float()
        # get object point cloud from dex_data according to observation space
        obj_pcl_2h = (
            cur_obs[:, :, eval_dataset.obs_state_dim : eval_dataset.obs_state_dim + eval_dataset.obs_pcl_dim]
            .clone()
            .to(device)
            .float()
            .reshape(1, args.n_obs_steps, pcl_number, 3)
        )
        obj_pcl_2h = obj_pcl_2h.permute(0, 1, 3, 2)
        robot_action = torch.randn(1, eval_dataset.horizon, eval_dataset.action_dim).to(device).float()

        # eval target
        if args.sampler == "ode":
            in_process_sample, res = cond_ode_sampler(
                score,
                prior_fn,
                sde_fn,
                (robot_action, obj_pcl_2h, state),
                t0=0.5,
                device=device,
                num_steps=1000,
                is_random=True,
                batch_size=1,
                hand_pcl=args.hand_pcl,
                full_state=None,
                envs=env,
                hand_model=None,
                relative=args.relative,
                action_type=args.action_type,
                space=args.space,
            )
            if eval_dataset.norm_data:
                print(
                    F.pairwise_distance(
                        denormalize(res[:, 0, :], dof_lower_limits, dof_upper_limits), abs_action[i].unsqueeze(0)
                    )
                )
                # test abs action denormalize(res[:, 0, :].to(torch.float32), dof_lower_limits, dof_upper_limits)
                set_states = denormalize(res[:, 0, :].to(torch.float32), dof_lower_limits, dof_upper_limits)
            else:
                print(F.pairwise_distance(res[:, 0, :], abs_action[i].unsqueeze(0)))
                set_states = res[:, 0, :].to(torch.float32)
            env.set_states(robot_dof=set_states, step_time=-1, set_dof_state=False, arm_ik=True)
            next_obs, rews, dones, infos = env.step(action[i : i + 1] * 0)
        elif args.sampler == "pc":
            st = time.time()
            res = pc_sampler(
                score,
                prior_fn,
                sde_fn,
                (robot_action, obj_pcl_2h, state),
                t0=0.5,
                device=device,
                num_steps=500,
                is_random=True,
                relative=args.relative,
                action_type=args.action_type,
                space=args.space,
            )
            print(time.time() - st)
            if eval_dataset.norm_data:
                print(
                    F.pairwise_distance(
                        denormalize(res[:, 0, :], dof_lower_limits, dof_upper_limits), abs_action[i].unsqueeze(0)
                    )
                )
                # test abs action denormalize(res[:, 0, :].to(torch.float32), dof_lower_limits, dof_upper_limits)
                set_states = denormalize(res[:, 0, :].to(torch.float32), dof_lower_limits, dof_upper_limits)
            else:
                print(F.pairwise_distance(res[:, 0, :], abs_action[i].unsqueeze(0)))
                set_states = res[:, 0, :].to(torch.float32)
            env.set_states(robot_dof=set_states, step_time=-1, set_dof_state=False, arm_ik=True)
            next_obs, rews, dones, infos = env.step(action[i : i + 1] * 0)
        elif args.sampler == "grad":
            # TODO real obs, action[0] = action[1], action[1:] = new ?
            abs_arm_tran_action = env.target_eef_pos.clone()
            abs_arm_rot_action = env.target_eef_euler.clone()
            abs_hand_action = env.curr_targets.clone()[:, env.hand_actuated_dof_indices]
            abs_action = torch.cat([abs_arm_tran_action, abs_arm_rot_action, abs_hand_action], -1)
            robot_action = abs_action.unsqueeze(0).repeat(1, eval_dataset.horizon, 1)
            batch_time_step = torch.ones(1, device=device).unsqueeze(1) * args.t0
            if args.space == "riemann":
                score_action = action2grad(robot_action, action_type=args.action_type)

            if real_obs:
                state = current_obs[:, :, : eval_dataset.obs_state_dim].clone().to(device).float()
                obj_pcl_2h = (
                    current_obs[
                        :, :, eval_dataset.obs_state_dim : eval_dataset.obs_state_dim + eval_dataset.obs_pcl_dim
                    ]
                    .clone()
                    .to(device)
                    .float()
                    .reshape(env.num_envs, eval_dataset.n_obs_steps, pcl_number, 3)
                )
                obj_pcl_2h = obj_pcl_2h.permute(0, 1, 3, 2)
            grad = score((score_action, obj_pcl_2h, state), batch_time_step.unsqueeze(1))
            if args.space == "riemann":
                grad = action2grad(grad, inv=True, action_type=args.action_type)

            # TODO demo obs , start = 0 ?
            start = eval_dataset.n_obs_steps - 1
            end = start + 1
            step_actions = grad[:, start, :]
            next_obs, rews, dones, infos = env.step(step_actions)

            if real_obs:
                current_obs.copy_(next_obs["obs"].reshape(env.num_envs, eval_dataset.n_obs_steps, -1))

        if args.web_visualizer_port != -1:
            # TODO enable for multiple env
            visualizer.set_cam_pose([0.8, 0, 1.2], [0, 0, 0.5])
            imgs.append(visualizer.render())

    if args.web_visualizer_port != -1:
        save_video(
            imgs=imgs,
            save_path=os.path.join(video_path, f"expert_wobj"),
            suffix="mp4",
        )

        save_video(
            imgs=imgs,
            save_path=os.path.join(video_path, f"expert_wobj"),
            suffix="gif",
        )

    print(f"rot_dist:{env.rot_dist.item()} pos_dist:{env.pos_dist.item()} fj_dist:{env.fj_dist.item()}")


def eval_predict(
    eval_dataset, eval_dataloader, score, epoch, device, env, args, prior_fn=None, sde_fn=None, score_mode="SDE"
):
    batch_predict_error = []
    step_predict_error = []
    for epoch in tqdm.trange(1):
        for i, dex_data in enumerate(eval_dataloader):
            if i > 50:
                break
            cur_step = i + epoch * len(eval_dataloader)

            batch_size = dex_data[0].size(0)

            obs = dex_data[0]
            action = dex_data[1]

            state = obs[:, :, : eval_dataset.obs_state_dim].clone().to(device).float()
            # get object point cloud from dex_data according to observation space
            obj_pcl_2h = (
                obs[:, :, eval_dataset.obs_state_dim : eval_dataset.obs_state_dim + eval_dataset.obs_pcl_dim]
                .clone()
                .to(device)
                .float()
                .reshape(batch_size, eval_dataset.horizon, pcl_number, 3)
            )
            obj_pcl_2h = obj_pcl_2h.permute(0, 1, 3, 2)
            if args.cond_on_arm:
                if args.action_type == "hand":
                    arm_action = action[:, :, :6].clone().to(device).float()
                    hand_action = action[:, :, 6:26].clone().to(device).float()
                    robot_action = hand_action.clone().to(device).float()
                elif args.action_type == "all":
                    arm_action = action[:, :, :6].clone().to(device).float()
                    robot_action = action.clone().to(device).float()
                state = torch.cat((state, arm_action), dim=-1)
            else:
                robot_action = action.clone().to(device).float()

            if args.dual_score:
                if args.cond_on_arm and (args.action_type == "hand" or args.action_type == "all"):
                    arm_state = state[:, :, :-6].clone().to(device).float()
                else:
                    arm_state = state.clone().to(device).float()
                hand_state = state.clone().to(device).float()
                in_process_sample_arm, res_arm = cond_ode_sampler(
                    score["arm"],
                    prior_fn,
                    sde_fn,
                    (robot_action[:, :, :6], obj_pcl_2h, arm_state),  # magic number
                    t0=0.5,
                    device=device,
                    num_steps=100,
                    is_random=True,
                    batch_size=batch_size,
                    hand_pcl=args.hand_pcl,
                    full_state=None,
                    envs=env,
                    hand_model=None,
                    relative=args.relative,
                    action_type=args.action_type,
                    space=args.space,
                )
                in_process_sample_hand, res_hand = cond_ode_sampler(
                    score["hand"],
                    prior_fn,
                    sde_fn,
                    (robot_action[:, :, 6:26], obj_pcl_2h, hand_state),  # magic number
                    t0=0.5,
                    device=device,
                    num_steps=100,
                    is_random=True,
                    batch_size=batch_size,
                    hand_pcl=args.hand_pcl,
                    full_state=None,
                    envs=env,
                    hand_model=None,
                    relative=args.relative,
                    action_type=args.action_type,
                    space=args.space,
                )
                res = torch.cat([res_arm, res_hand], -1)
            else:
                if score_mode == "SDE":
                    # if args.sampler == "ode":
                    #     in_process_sample, res = cond_ode_sampler(
                    #         score,
                    #         prior_fn,
                    #         sde_fn,
                    #         (robot_action, obj_pcl_2h, state),
                    #         t0=0.5,
                    #         device=device,
                    #         num_steps=1000,
                    #         is_random=True,
                    #         batch_size=batch_size,
                    #         hand_pcl=args.hand_pcl,
                    #         full_state=None,
                    #         envs=env,
                    #         hand_model=None,
                    #         relative=args.relative,
                    #         action_type=args.action_type,
                    #         space=args.space,
                    #     )
                    # elif args.sampler == "pc":
                    res = pc_sampler(
                        score,
                        prior_fn,
                        sde_fn,
                        (robot_action, obj_pcl_2h, state),
                        t0=0.5,
                        device=device,
                        num_steps=500,
                        is_random=True,
                        relative=args.relative,
                        action_type=args.action_type,
                        space=args.space,
                    )
                elif score_mode == "DDIM":
                    res = score.predict_action((robot_action, obj_pcl_2h, state))

            if args.action_mode == "abs" or args.action_mode == "obs":
                if eval_dataset.norm_data:
                    robot_action = denormalize(
                        robot_action, eval_dataset.dof_lower_limits, eval_dataset.dof_upper_limits
                    )
                    res = denormalize(res, eval_dataset.dof_lower_limits, eval_dataset.dof_upper_limits)
                robot_action[:, :, 3:6] = torch.arccos(torch.cos(robot_action[:, :, 3:6]))
                res[:, :, 3:6] = torch.arccos(torch.cos(res[:, :, 3:6]))
            batch_predict_error.append(torch.mean(torch.sum(F.pairwise_distance(robot_action, res), -1)))
            step_predict_error.append(torch.mean(F.pairwise_distance(robot_action, res)))
    return torch.mean(torch.stack(batch_predict_error)), torch.mean(torch.stack(step_predict_error))


class PgmDataset(Dataset):
    def __init__(self, dataset_path, args=None, norm_data=False):
        if "train" in dataset_path:
            demo_num = args.demo_nums
        else:
            demo_num = None
        self.action_mode = args.action_mode
        self.action_type = args.action_type

        self.dataset_path = dataset_path

        self.env_mode = None
        self.obs_space = None
        self.action_space = None
        self.action_range = None
        self.predefined_object_codes = []
        self.dataset = []

        self.n_obs_steps = args.n_obs_steps
        self.n_action_steps = args.n_action_steps
        self.horizon = args.n_prediction_steps

        self.cond_on_arm = args.cond_on_arm

        # TODO get from saved dataset magic number, currently not have tactile as obs
        self.obs_tactile_dim = 14
        self.fingertip_from_all = False
        if self.fingertip_from_all:
            self.obs_tactile_dim = 5
            self.full_obs_tactile_dim = 14
        self.obs_state_dim = 208 + self.obs_tactile_dim
        self.pcl_number = pcl_number
        self.obs_pcl_dim = self.pcl_number * 3
        self.pcl_type = None
        self.ft_idx_in_all = None

        self.full_action_dim = 26
        self.arm_action_dim = 6
        self.hand_action_dim = 20

        self.dof_lower_limits = None
        self.dof_upper_limits = None

        self.norm_data = norm_data

        # action type: arm, hand, all
        if self.action_type == "arm":
            self.action_dim = 6
        elif self.action_type == "hand":
            self.action_dim = 20
        elif self.action_type == "all":
            self.action_dim = 26

        filepaths = [Path(dataset_path) / filename for filename in os.listdir(dataset_path)]
        trajectory_lengths = []

        for filepath in tqdm.tqdm(filepaths, desc="loading dataset"):
            obs, action = self.prase_trajectory(filepath)
            trajectory_lengths.append(obs.shape[0])

        self.filepaths = filepaths
        self.num_samples = sum([i - self.horizon + 1 for i in trajectory_lengths])
        self.trajectory_lengths = torch.tensor(trajectory_lengths, dtype=torch.long)
        self.cumsum_trajectory_lengths = torch.cumsum(self.trajectory_lengths - self.horizon + 1, dim=0)
        self.trajectory_indices = torch.cat(
            [
                torch.ones(length - self.horizon + 1, dtype=torch.long) * i
                for i, length in enumerate(self.trajectory_lengths)
            ],
            dim=0,
        )
        self.trajectory_lengths = self.trajectory_lengths.cpu().numpy()
        self.cumsum_trajectory_lengths = self.cumsum_trajectory_lengths.cpu().numpy()
        self.trajectory_indices = self.trajectory_indices.cpu().numpy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        # binary search trajectory index
        trajectory_index = self.trajectory_indices[index]

        obs, action = self.prase_trajectory(self.filepaths[trajectory_index])
        obs = torch.from_numpy(obs)
        action = torch.from_numpy(action)
        index = index - self.cumsum_trajectory_lengths[trajectory_index - 1] if trajectory_index > 0 else index

        obs = obs[index : index + self.horizon]
        action = action[index : index + self.horizon]
        return obs, action

    def prase_trajectory(self, filepath: os.PathLike):
        demo = np.load(filepath, allow_pickle=True).tolist()
        if self.obs_space is None:
            self.pcl_type = demo["pcl_type"]
            self.obs_space = demo["obs_space"]
            if "tactile" not in self.obs_space:
                self.obs_space.append("tactile")

            # if self.pcl_type == "full_pcl" and "pointcloud_wrt_palm" not in self.obs_space:
            #     self.obs_space.append("pointcloud_wrt_palm")
            # elif self.pcl_type == "real_pcl" and "rendered_pointcloud" not in self.obs_space:
            #     self.obs_space.append("rendered_pointcloud")
            self.obs_space.append("pointcloud_wrt_palm")
            self.obs_space.append("imagined_pointcloud")

            self.ft_idx_in_all = demo["ft_idx_in_all"]
            self.action_space = demo["action_space"]
            self.env_mode = demo["env_mode"]
            self.action_range = demo["dof_upper_limit"] - demo["dof_lower_limit"]
            self.dof_lower_limits = torch.tensor(demo["dof_lower_limit"], dtype=torch.float32)
            self.dof_upper_limits = torch.tensor(demo["dof_upper_limit"], dtype=torch.float32)

        object_code = demo["object_code"]
        if object_code not in self.predefined_object_codes:
            self.predefined_object_codes.append(object_code)

        obs = demo["obs"]

        if self.fingertip_from_all:
            # reorginize obs
            state_obs = obs[:, : self.obs_state_dim - self.obs_tactile_dim]  # magic number
            tactile_obs = obs[
                :,
                self.obs_state_dim
                - self.obs_tactile_dim : self.obs_state_dim
                - self.obs_tactile_dim
                + self.full_obs_tactile_dim,
            ]
            pcl_obs = obs[:, self.obs_state_dim - self.obs_tactile_dim + self.full_obs_tactile_dim :]
            assert pcl_obs.shape[1] == self.obs_pcl_dim

            # get fingertip obs from all obs
            tactile_obs = tactile_obs[:, self.ft_idx_in_all]

            obs = np.concatenate([state_obs, tactile_obs, pcl_obs], axis=1)
        # obs = demo["obs"]
        # print(obs.shape[1], self.obs_state_dim, self.obs_pcl_dim)
        assert obs.shape[1] == self.obs_state_dim + self.obs_pcl_dim
        # action mode: rel, abs, obs
        if self.action_mode == "rel":
            action = demo["action"]
        elif self.action_mode == "abs":
            if self.norm_data:
                action = normalize(demo["abs_action"], demo["dof_lower_limit"], demo["dof_upper_limit"])
            else:
                action = demo["abs_action"]
        elif self.action_mode == "obs":
            if self.norm_data:
                action = normalize(demo["obs_action"], demo["dof_lower_limit"], demo["dof_upper_limit"])
            else:
                action = demo["obs_action"]

        if self.action_type == "arm":
            action = action[:, : self.action_dim]
        elif self.action_type == "hand" and not self.cond_on_arm:
            action = action[:, 6 : 6 + self.action_dim]

        # action = demo['action']
        return obs, action


class IladDataset(Dataset):
    def __init__(self, dataset_path, action_mode="rel", action_type="all", obs_info=None, norm_data=False):
        self.action_mode = action_mode
        self.action_type = action_type

        self.dataset_path = dataset_path

        self.env_mode = None
        self.obs_space = None
        self.action_space = None
        self.action_range = None
        self.predefined_object_codes = []
        self.dataset = []

        self.horizon = 1
        self.obs_info = obs_info

        # TODO get from saved dataset magic number, currently not have tactile as obs
        self.obs_tactile_dim = 14
        self.fingertip_from_all = False
        if self.fingertip_from_all:
            self.obs_tactile_dim = 5
            self.full_obs_tactile_dim = 14
        self.obs_state_dim = 208 + self.obs_tactile_dim
        self.pcl_number = pcl_number
        self.obs_pcl_dim = self.pcl_number * 3
        self.pcl_type = None
        self.ft_idx_in_all = None

        self.full_action_dim = 26
        self.arm_action_dim = 6
        self.hand_action_dim = 20

        self.dof_lower_limits = None
        self.dof_upper_limits = None

        self.norm_data = norm_data

        # action type: arm, hand, all
        if self.action_type == "arm":
            self.action_dim = 6
        elif self.action_type == "hand":
            self.action_dim = 20
        elif self.action_type == "all":
            self.action_dim = 26

        filepaths = [Path(dataset_path) / filename for filename in os.listdir(dataset_path)]
        trajectory_lengths = []

        for filepath in tqdm.tqdm(filepaths, desc="loading dataset"):
            obs, action = self.prase_trajectory(filepath)
            trajectory_lengths.append(obs.shape[0])

        self.filepaths = filepaths
        self.num_samples = sum([i - self.horizon + 1 for i in trajectory_lengths])
        self.trajectory_lengths = torch.tensor(trajectory_lengths, dtype=torch.long)
        self.cumsum_trajectory_lengths = torch.cumsum(self.trajectory_lengths - self.horizon + 1, dim=0)
        self.trajectory_indices = torch.cat(
            [
                torch.ones(length - self.horizon + 1, dtype=torch.long) * i
                for i, length in enumerate(self.trajectory_lengths)
            ],
            dim=0,
        )
        self.trajectory_lengths = self.trajectory_lengths.cpu().numpy()
        self.cumsum_trajectory_lengths = self.cumsum_trajectory_lengths.cpu().numpy()
        self.trajectory_indices = self.trajectory_indices.cpu().numpy()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int):
        # binary search trajectory index
        trajectory_index = self.trajectory_indices[index]

        obs, action = self.prase_trajectory(self.filepaths[trajectory_index])
        obs = torch.from_numpy(obs)
        action = torch.from_numpy(action)
        index = index - self.cumsum_trajectory_lengths[trajectory_index - 1] if trajectory_index > 0 else index

        obs = obs[index]
        action = action[index]
        return obs, action

    def get_random_tuple(self, indices):
        obs_all = torch.tensor([])
        action_all = torch.tensor([])
        for index in indices:
            obs, action = self.__getitem__(index)
            obs_all = torch.cat([obs_all, obs.unsqueeze(0)])
            action_all = torch.cat([action_all, action.unsqueeze(0)])
        return obs_all, action_all

    def collate_fn(self, batch):
        obs, action = zip(*batch)
        obs = torch.cat(obs, dim=0)
        action = torch.cat(action, dim=0)
        return obs, action

    def fetch_state_from_obs_info(self, obs):
        self.encode_state_dim = 0
        for info in self.obs_info:
            if info["name"] == "ur_endeffector_position":
                encode_state_obs = obs[:, :7]
                self.encode_state_dim = self.encode_state_dim + 7
            if info["name"] == "shadow_hand_dof_position":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 7:37]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 30
            if info["name"] == "object_position_wrt_palm":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 132:139]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 7
            if info["name"] == "object_target_relposecontact":
                encode_state_obs = np.concatenate([encode_state_obs, obs[:, 152:177]], axis=-1)
                self.encode_state_dim = self.encode_state_dim + 25
        return encode_state_obs

    def prase_trajectory(self, filepath: os.PathLike):
        demo = np.load(filepath, allow_pickle=True).tolist()
        if self.obs_space is None:
            self.pcl_type = demo["pcl_type"]
            self.obs_space = demo["obs_space"]
            if "tactile" not in self.obs_space:
                self.obs_space.append("tactile")

            if self.pcl_type == "full_pcl" and "pointcloud_wrt_palm" not in self.obs_space:
                self.obs_space.append("pointcloud_wrt_palm")
            elif self.pcl_type == "real_pcl" and "rendered_pointcloud" not in self.obs_space:
                self.obs_space.append("rendered_pointcloud")

            self.ft_idx_in_all = demo["ft_idx_in_all"]
            self.action_space = demo["action_space"]
            self.env_mode = demo["env_mode"]
            self.action_range = demo["dof_upper_limit"] - demo["dof_lower_limit"]
            self.dof_lower_limits = torch.tensor(demo["dof_lower_limit"], dtype=torch.float32)
            self.dof_upper_limits = torch.tensor(demo["dof_upper_limit"], dtype=torch.float32)

        object_code = demo["object_code"]
        if object_code not in self.predefined_object_codes:
            self.predefined_object_codes.append(object_code)

        obs = demo["obs"]

        if self.fingertip_from_all:
            # reorginize obs
            state_obs = self.fetch_state_from_obs_info(
                obs[:, : self.obs_state_dim - self.obs_tactile_dim]
            )  # magic number
            tactile_obs = obs[
                :,
                self.obs_state_dim
                - self.obs_tactile_dim : self.obs_state_dim
                - self.obs_tactile_dim
                + self.full_obs_tactile_dim,
            ]
            pcl_obs = obs[:, self.obs_state_dim - self.obs_tactile_dim + self.full_obs_tactile_dim :]
            assert pcl_obs.shape[1] == self.obs_pcl_dim

            # get fingertip obs from all obs
            tactile_obs = tactile_obs[:, self.ft_idx_in_all]

            obs = np.concatenate([state_obs, tactile_obs, pcl_obs], axis=1)
        else:
            state_obs = self.fetch_state_from_obs_info(
                obs[:, : self.obs_state_dim - self.obs_tactile_dim]
            )  # magic number
            other_obs = obs[:, self.obs_state_dim - self.obs_tactile_dim :]
            obs = np.concatenate([state_obs, other_obs], axis=1)

        # obs = demo["obs"]
        # print(obs.shape[1], self.obs_state_dim, self.obs_pcl_dim)
        assert obs.shape[1] == self.encode_state_dim + self.obs_pcl_dim + self.obs_tactile_dim
        # action mode: rel, abs, obs
        if self.action_mode == "rel":
            action = demo["action"]
        elif self.action_mode == "abs":
            if self.norm_data:
                action = normalize(demo["abs_action"], demo["dof_lower_limit"], demo["dof_upper_limit"])
            else:
                action = demo["abs_action"]
        elif self.action_mode == "obs":
            if self.norm_data:
                action = normalize(demo["obs_action"], demo["dof_lower_limit"], demo["dof_upper_limit"])
            else:
                action = demo["obs_action"]

        return obs, action
