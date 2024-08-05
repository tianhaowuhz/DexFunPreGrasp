from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from isaacgym.torch_utils import quat_from_euler_xyz, quat_mul, quat_conjugate


@torch.jit.script
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


@torch.jit.script
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


@torch.jit.script
def saturate(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Clamps a given input tensor to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Clamped transform of the tensor. Shape (N, dims)
    """
    return torch.max(torch.min(x, upper), lower)


@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns sampled uniform positions in circle
    (https://stackoverflow.com/a/50746409)"""
    # sample radius of circle
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    # sample theta of point
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    # x,y-position of the cube
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: torch.device) -> torch.Tensor:
    """Returns sampled height of the goal object."""
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns identity rotation transform."""
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation in 3D as quaternion.

    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
    """
    # sample random orientation from normal distribution
    quat = torch.randn((num, 4), dtype=torch.float, device=device)
    # normalize the quaternion
    quat = F.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat


@torch.jit.script
def random_orientation_within_angle(num: int, device: torch.device, base: torch.Tensor, max_angle: float):
    """Generates random quaternions within max_angle of base
    Ref: https://math.stackexchange.com/a/3448434
    """
    quat = torch.zeros((num, 4), dtype=torch.float, device=device)

    rand = torch.rand((num, 3), dtype=torch.float, device=device)

    c = torch.cos(rand[:, 0] * max_angle)
    n = torch.sqrt((1.0 - c) / 2.0)

    quat[:, 3] = torch.sqrt((1 + c) / 2.0)
    quat[:, 2] = (rand[:, 1] * 2.0 - 1.0) * n
    quat[:, 0] = (torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.cos(2 * np.pi * rand[:, 2])) * n
    quat[:, 1] = (torch.sqrt(1 - quat[:, 2] ** 2.0) * torch.sin(2 * np.pi * rand[:, 2])) * n

    # floating point errors can cause it to  be slightly off, re-normalise
    quat = F.normalize(quat, p=2.0, dim=-1, eps=1e-12)

    return quat_mul(quat, base)


@torch.jit.script
def random_angular_vel(num: int, device: torch.device, magnitude_stdev: float) -> torch.Tensor:
    """Samples a random angular velocity with standard deviation `magnitude_stdev`"""

    axis = torch.randn((num, 3), dtype=torch.float, device=device)
    axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
    magnitude = torch.randn((num, 1), dtype=torch.float, device=device)
    magnitude *= magnitude_stdev
    return magnitude * axis


@torch.jit.script
def random_yaw_orientation(num: int, device: torch.device) -> torch.Tensor:
    """Returns sampled rotation around z-axis."""
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)

@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]
    
def multiply_transform(s_pos, s_quat, t_pos, t_quat):
    t2s_quat = quat_mul(s_quat, t_quat)

    B = t_pos.size()[0]
    padding = torch.zeros([B, 1]).to(t_pos.device)
    t_pos_pad = torch.cat([t_pos,padding],-1)
    s_quat = s_quat.expand_as(t_quat)
    t2s_pos = transform_points(s_quat, t_pos_pad)
    s_pos = s_pos.expand_as(t2s_pos)
    t2s_pos += s_pos
    return t2s_pos, t2s_quat