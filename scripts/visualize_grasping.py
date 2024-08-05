import argparse
import json
import os

import isaacgym
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import quat_conjugate, quat_mul


@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:, :, :3]
    elif len(pt_new.size()) == 2:
        return pt_new[:, :3]


def multiply_transform(s_pos, s_quat, t_pos, t_quat):
    t2s_quat = quat_mul(s_quat, t_quat)

    B = t_pos.size()[0]
    padding = torch.zeros([B, 1]).to(t_pos.device)
    t_pos_pad = torch.cat([t_pos, padding], -1)
    s_quat = s_quat.expand_as(t_quat)
    t2s_pos = transform_points(s_quat, t_pos_pad)
    s_pos = s_pos.expand_as(t2s_pos)
    t2s_pos += s_pos
    return t2s_pos, t2s_quat


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--meshdata", default="assets/oakink", type=str)
parser.add_argument("--dataset", default="data/oakink_shadow_dataset", type=str)
parser.add_argument(
    "--grasp_data",
    default="data/oakink_shadow_dataset_valid_force_noise_accept_1/cylinder_bottle_s552/006226_liftup_s15552.json",
    type=str,
)
parser.add_argument("--hand_root", default="assets/shadow_robot", type=str)
parser.add_argument("--hand_file", default="shadow_hand_right.urdf", type=str)

args = parser.parse_args()

###################################################
# parse data
###################################################

meshpath = args.meshdata
datapath = args.dataset
gpu = args.gpu
vis_object_pose_wrt_palm = True

translation_names = ["x", "y", "z"]
rot_names = ["x", "y", "z", "w"]
joint_names = [
    "rh_WRJ2",
    "rh_WRJ1",
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_FFJ2",
    "rh_FFJ1",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_MFJ2",
    "rh_MFJ1",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_RFJ2",
    "rh_RFJ1",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_LFJ2",
    "rh_LFJ1",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
]

with open(args.grasp_data) as fr:
    grasp_data = json.load(fr)

joint = grasp_data["joints"]
hand_pos = [0, 0, 0]
hand_quat = [0, 0, 0, 1]

hand_qpos = {name: joint[name] for name in joint_names}

if vis_object_pose_wrt_palm:
    object_position = grasp_data["object_pose_wrt_palm"]["position"]
    object_quaternion = grasp_data["object_pose_wrt_palm"]["quaternion"]
    object_pos = torch.tensor([object_position[name] for name in translation_names]).reshape(-1, 3)
    object_quat = torch.tensor([object_quaternion[name] for name in rot_names]).reshape(-1, 4)
else:
    object_position = grasp_data["object_pose_wrt_forearm"]["position"]
    object_quaternion = grasp_data["object_pose_wrt_forearm"]["quaternion"]
    object_pos = [object_position[name] for name in translation_names]
    object_quat = [object_quaternion[name] for name in rot_names]

obj_scale = 1
###################################################


###################################################
# setting isaac
###################################################

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0
sim_params.use_gpu_pipeline = False
sim = gym.create_sim(gpu, gpu, gymapi.SIM_PHYSX, sim_params)

camera_props = gymapi.CameraProperties()
camera_props.width = 800
camera_props.height = 600
# camera_props.use_collision_geometry = True

###################################################

###################################################
# setting asserts
###################################################
object_asset_options = gymapi.AssetOptions()
object_asset_options.density = 1000.0
object_asset_options.convex_decomposition_from_submeshes = True
object_asset_options.override_com = True
object_asset_options.override_inertia = True
obj_code = grasp_data["object_code"]
obj_asset_root = os.path.join(meshpath, obj_code)
obj_asset_file = "decomposed.urdf"
obj_asset = gym.load_asset(sim, obj_asset_root, obj_asset_file, object_asset_options)

hand_asset_root = args.hand_root
hand_asset_file = args.hand_file
hand_asset_options = gymapi.AssetOptions()
hand_asset_options.disable_gravity = True
hand_asset_options.fix_base_link = True
hand_asset_options.collapse_fixed_joints = True
hand_asset = gym.load_asset(sim, hand_asset_root, hand_asset_file, hand_asset_options)

###################################################

###################################################
# creat env
###################################################

env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 6)

pose = gymapi.Transform()
pose.r = gymapi.Quat(*hand_quat)
pose.p = gymapi.Vec3(*hand_pos)
shand = gym.create_actor(env, hand_asset, pose, "shand", 0, -1)
hand_actor_handle = gym.get_actor_handle(env, shand)
hand_props = gym.get_actor_dof_properties(env, hand_actor_handle)
hand_props["driveMode"].fill(gymapi.DOF_MODE_POS)
hand_props["stiffness"].fill(1000)
hand_props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, hand_actor_handle, hand_props)
dof_states = gym.get_actor_dof_states(env, hand_actor_handle, gymapi.STATE_ALL)
for name in joint_names:
    joint_idx = gym.find_actor_dof_index(env, hand_actor_handle, name, gymapi.DOMAIN_ACTOR)
    dof_states["pos"][joint_idx] = hand_qpos[name]
gym.set_actor_dof_states(env, hand_actor_handle, dof_states, gymapi.STATE_ALL)
gym.set_actor_dof_position_targets(env, hand_actor_handle, dof_states["pos"])
hand_shape_props = gym.get_actor_rigid_shape_properties(env, hand_actor_handle)
for i in range(len(hand_shape_props)):
    hand_shape_props[i].friction = 3
gym.set_actor_rigid_shape_properties(env, hand_actor_handle, hand_shape_props)

if vis_object_pose_wrt_palm:
    rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
    hand_palm_handle = gym.find_asset_rigid_body_index(hand_asset, "rh_palm")
    hand_palm_handle = torch.tensor(hand_palm_handle, dtype=torch.long)
    hand_palm_pose = rigid_body_states[:, hand_palm_handle][:, :7]
    object_pos, object_quat = multiply_transform(hand_palm_pose[:, :3], hand_palm_pose[:, 3:7], object_pos, object_quat)
    object_pos = object_pos.squeeze().cpu().numpy()
    object_quat = object_quat.squeeze().cpu().numpy()

pose = gymapi.Transform()
pose.p = gymapi.Vec3(*object_pos)
pose.r = gymapi.Quat(*object_quat)
obj = gym.create_actor(env, obj_asset, pose, "obj", 0, 1)
obj_actor_handle = gym.get_actor_handle(env, obj)
gym.set_actor_scale(env, obj_actor_handle, obj_scale)
obj_shape_props = gym.get_actor_rigid_shape_properties(env, obj_actor_handle)
for i in range(len(obj_shape_props)):
    obj_shape_props[i].friction = 3
gym.set_actor_rigid_shape_properties(env, obj_actor_handle, obj_shape_props)
###################################################

###################################################
# viewer
###################################################
viewer = gym.create_viewer(sim, camera_props)
gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(0, 0, 1), gymapi.Vec3(0, 0, 0))
###################################################

###################################################
# sim loop
###################################################
while not gym.query_viewer_has_closed(viewer):
    # gym.simulate(sim)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
###################################################
