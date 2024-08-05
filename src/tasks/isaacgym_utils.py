from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply, quat_conjugate, quat_mul
from torch import Tensor


class ObservationSpec(object):
    def __init__(
        self,
        name: str,
        dim: int,
        attr: str,
        shape: Optional[Tuple[int, ...]] = None,
        tags: Optional[Sequence[str]] = None,
    ):
        self.name = name
        self.dim = dim
        self.attr = attr
        self.shape = shape if shape is not None else (dim,)
        self.tags = tags if tags is not None else []


class ActionSpec(object):
    def __init__(
        self,
        name: str,
        dim: int,
        attr: str,
        shape: Optional[Tuple[int, ...]] = None,
        tags: Optional[Sequence[str]] = None,
    ):
        self.name = name
        self.dim = dim
        self.attr = attr
        self.shape = shape if shape is not None else (dim,)
        self.tags = tags if tags is not None else []


def pack_pointcloud_observations(
    pointclouds_with_properties: Dict, mask: bool = True, device: torch.device = "cpu"
) -> torch.Tensor:
    """Pack pointclouds with properties into a single tensor.

    Args:
        pointclouds_with_properties (Dict): Dictionary of pointclouds with properties.
        {
            "synthetic": {
                "points": torch.Tensor, # shape (batch_size, num_points, 3)
                "properties": {
                    "finger": torch.Tensor, # shape (batch_size, num_points)
                    "contact": torch.Tensor, # shape (batch_size, num_points)
                }
            }, "rendered": {
                "points": torch.Tensor, # shape (batch_size, num_points, 3)
                "properties": {
                    "segmentation": torch.Tensor, # shape (batch_size, num_points)
                }
            }
        }

        mask (bool, optional): Category mask. Defaults to True.
        device (torch.device, optional): Device to put the tensor on. Defaults to "cpu".

    Returns:
        torch.Tensor: Input tensor for the pointcloud encoder. shape (batch_size, num_channels, num_points)
    """
    num_channels = 0
    num_points = 0
    num_properties = 0
    num_categories = 0
    batch_size = 0

    names = []
    prop_names = []
    for name in pointclouds_with_properties:
        names.append(name)
        prop_names.extend(list(pointclouds_with_properties[name].get("properties", {}).keys()))
        points = pointclouds_with_properties[name]["points"]
        num_points += points.shape[1]

        if batch_size == 0:
            batch_size = points.shape[0]

    names = sorted(names)
    prop_names = sorted(list(set(prop_names)))

    num_categories = len(names)
    num_properties = len(prop_names)
    num_channels = 3 + num_properties + (num_categories - 1 if mask else 0)

    buffer = torch.zeros((batch_size, num_channels, num_points), device=device)
    offset = 0
    for i, name in enumerate(names):
        points: torch.Tensor = pointclouds_with_properties[name]["points"]
        properties: Dict = pointclouds_with_properties[name].get("properties", {})

        n = points.size(1)

        buffer[:, :3, offset : offset + n] = points.transpose(1, 2)
        for prop_name in properties:
            buffer[:, 3 + prop_names.index(prop_name), offset : offset + n] = properties[prop_name]

        if mask and i > 0:
            buffer[:, 3 + num_properties + i - 1, offset : offset + n] = 1.0

        offset += n

    return buffer


def ik(
    jacobian_end_effector: torch.Tensor,
    current_position: torch.Tensor,
    current_orientation: torch.Tensor,
    goal_position: torch.Tensor,
    goal_orientation: Optional[torch.Tensor] = None,
    damping_factor: float = 0.05,
    squeeze_output: bool = True,
) -> torch.Tensor:
    """Inverse kinematics using damped least squares method.

    Borrowed from skrl.utils.isaacgym_utils (skrl v0.10.2)

    Args:
        jacobian_end_effector (torch.Tensor): End effector's jacobian
        current_position (torch.Tensor): End effector's current position
        current_orientation (torch.Tensor): End effector's current orientation
        goal_position (torch.Tensor): End effector's goal position
        goal_orientation (torch.Tensor, optional): End effector's goal orientation (default: None)
        damping_factor (float, optional): Damping factor (default: 0.05)
        squeeze_output (bool, optional): Squeeze output (default: True)

    Returns:
        torch.Tensor: Change in joint angles
    """
    if goal_orientation is None:
        goal_orientation = current_orientation

    # compute error
    q = quat_mul(goal_orientation, quat_conjugate(current_orientation))
    error = torch.cat(
        [
            goal_position - current_position,  # position error
            q[:, 0:3] * torch.sign(q[:, 3]).unsqueeze(-1),  # orientation error
        ],
        dim=-1,
    ).unsqueeze(-1)

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6, device=jacobian_end_effector.device) * (damping_factor**2)
    if squeeze_output:
        return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error).squeeze(dim=2)
    else:
        return transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ error


def position(transform: gymapi.Transform, device: torch.device = "cpu") -> Tensor:
    """Get the position of a transform.

    Args:
        transform (gymapi.Transform): Isaac Gym Transform instance
        device (torch.device): Device to put the tensor on

    Returns:
        Tensor: shape (3,)
    """
    return torch.tensor([transform.p.x, transform.p.y, transform.p.z], device=device)


def orientation(transform: gymapi.Transform, device: torch.device = "cpu") -> Tensor:
    """Get the orientation of a transform.

    Args:
        transform (gymapi.Transform): Isaac Gym Transform instance
        device (torch.device): Device to put the tensor on

    Returns:
        Tensor: shape (4,)
    """
    return torch.tensor([transform.r.x, transform.r.y, transform.r.z, transform.r.w], device=device)


def draw_axes(
    gym: gymapi.Gym,
    viewer: gymapi.Viewer,
    envs: Sequence[gymapi.Env],
    positions: Tensor,
    orientations: Tensor,
    length: float = 0.5,
) -> None:
    """Draw axes at the given positions and orientations.

    Args:
        gym (gymapi.Gym): Isaac Gym instance
        viewer (gymapi.Viewer): Isaac Gym viewer instance
        envs (Sequence[gymapi.Env]): List of Isaac Gym environments
        positions (Tensor): shape (num_envs, 3)
        orientations (Tensor): shape (num_envs, 4)
        length (float, optional): length of the axes. Defaults to 0.5.
    """
    assert (positions.ndim == 2 or positions.ndim == 3) and positions.shape[-1] == 3
    assert (orientations.ndim == 2 or orientations.ndim == 3) and orientations.shape[-1] == 4
    assert positions.shape[0] == orientations.shape[0] == len(envs)
    assert positions.device == orientations.device, "positions and orientations must be on the same device"

    num_envs = positions.shape[0]
    num_lines = positions.shape[1] if positions.ndim == 3 else 1
    device = positions.device

    x_unit = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(positions)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(positions)
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(positions)

    x = (positions + quat_apply(orientations, x_unit) * length).detach().cpu().numpy()
    y = (positions + quat_apply(orientations, y_unit) * length).detach().cpu().numpy()
    z = (positions + quat_apply(orientations, z_unit) * length).detach().cpu().numpy()

    r = torch.tensor([1.0, 0.0, 0.0]).expand_as(positions).numpy()
    g = torch.tensor([0.0, 1.0, 0.0]).expand_as(positions).numpy()
    b = torch.tensor([0.0, 0.0, 1.0]).expand_as(positions).numpy()

    positions = positions.detach().cpu().numpy()

    for i in range(num_envs):
        gym.add_lines(viewer, envs[i], num_lines, np.concatenate([positions[i], x[i]], axis=-1), r[i])
        gym.add_lines(viewer, envs[i], num_lines, np.concatenate([positions[i], y[i]], axis=-1), g[i])
        gym.add_lines(viewer, envs[i], num_lines, np.concatenate([positions[i], z[i]], axis=-1), b[i])


def draw_boxes(
    gym: gymapi.Gym,
    viewer: gymapi.Viewer,
    envs: Sequence[gymapi.Env],
    positions: Tensor,
    orientations: Tensor,
    size: Union[float, Tuple[float, float, float]] = 0.5,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    shadow_density: int = 3,
):
    assert (positions.ndim == 2 or positions.ndim == 3) and positions.shape[-1] == 3
    assert (orientations.ndim == 2 or orientations.ndim == 3) and orientations.shape[-1] == 4
    assert positions.shape[0] == orientations.shape[0] == len(envs)
    assert positions.device == orientations.device, "positions and orientations must be on the same device"

    num_envs = positions.shape[0]
    num_boxes = positions.shape[1] if positions.ndim == 3 else 1
    device = positions.device

    positions = positions.reshape(num_envs, num_boxes, 3)
    orientations = orientations.reshape(num_envs, num_boxes, 4)

    if isinstance(size, float):
        size = (size, size, size)

    x_unit = torch.tensor([1.0, 0.0, 0.0], device=device)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=device)
    z_unit = torch.tensor([0.0, 0.0, 1.0], device=device)

    corners = torch.stack(
        [
            x_unit * size[0] * 0.5 + y_unit * size[1] * 0.5 + z_unit * size[2] * 0.5,
            x_unit * size[0] * 0.5 + y_unit * size[1] * 0.5 - z_unit * size[2] * 0.5,
            x_unit * size[0] * 0.5 - y_unit * size[1] * 0.5 + z_unit * size[2] * 0.5,
            x_unit * size[0] * 0.5 - y_unit * size[1] * 0.5 - z_unit * size[2] * 0.5,
            -x_unit * size[0] * 0.5 + y_unit * size[1] * 0.5 + z_unit * size[2] * 0.5,
            -x_unit * size[0] * 0.5 + y_unit * size[1] * 0.5 - z_unit * size[2] * 0.5,
            -x_unit * size[0] * 0.5 - y_unit * size[1] * 0.5 + z_unit * size[2] * 0.5,
            -x_unit * size[0] * 0.5 - y_unit * size[1] * 0.5 - z_unit * size[2] * 0.5,
        ]
    )

    positions = positions.reshape(num_envs, num_boxes, 1, 3).repeat(1, 1, 8, 1)
    orientations = orientations.reshape(num_envs, num_boxes, 1, 4).repeat(1, 1, 8, 1)
    corners = corners.reshape(1, 1, 8, 3).repeat(num_envs, num_boxes, 1, 1)

    corners = positions + quat_apply(orientations, corners)
    corners = corners.detach().cpu().numpy()

    skeleton = np.concatenate(
        [
            corners[:, :, [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]],
            corners[:, :, [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]],
        ],
        axis=-1,
    )

    if shadow_density > 0:
        n = shadow_density

        # 0 / 2 / 4 / 6
        shadow_xy_z_pos = np.concatenate([corners[:, :, 0], corners[:, :, 6]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 2], corners[:, :, 2]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 4], corners[:, :, 4]], axis=-1)
        shadow_xy_z_pos = np.concatenate(
            [
                np.stack(
                    [shadow_xy_z_pos * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_xy_z_pos * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        # 1 / 3 / 5 / 7
        shadow_xy_z_neg = np.concatenate([corners[:, :, 3], corners[:, :, 5]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 1], corners[:, :, 1]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 7], corners[:, :, 7]], axis=-1)
        shadow_xy_z_neg = np.concatenate(
            [
                np.stack(
                    [shadow_xy_z_neg * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_xy_z_neg * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        # 0 / 1 / 2 / 3
        shadow_yz_x_pos = np.concatenate([corners[:, :, 0], corners[:, :, 3]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 1], corners[:, :, 1]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 2], corners[:, :, 2]], axis=-1)
        shadow_yz_x_pos = np.concatenate(
            [
                np.stack(
                    [shadow_yz_x_pos * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_yz_x_pos * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        # 4 / 5 / 6 / 7
        shadow_yz_x_neg = np.concatenate([corners[:, :, 5], corners[:, :, 6]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 4], corners[:, :, 4]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 5], corners[:, :, 5]], axis=-1)
        shadow_yz_x_neg = np.concatenate(
            [
                np.stack(
                    [shadow_yz_x_neg * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_yz_x_neg * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        # 0 / 1 / 4 / 5
        shadow_xz_y_pos = np.concatenate([corners[:, :, 0], corners[:, :, 5]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 1], corners[:, :, 1]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 4], corners[:, :, 4]], axis=-1)
        shadow_xz_y_pos = np.concatenate(
            [
                np.stack(
                    [shadow_xz_y_pos * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_xz_y_pos * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        # 2 / 3 / 6 / 7
        shadow_xz_y_neg = np.concatenate([corners[:, :, 3], corners[:, :, 6]], axis=-1)
        corner0 = np.concatenate([corners[:, :, 2], corners[:, :, 2]], axis=-1)
        corner1 = np.concatenate([corners[:, :, 7], corners[:, :, 7]], axis=-1)
        shadow_xz_y_neg = np.concatenate(
            [
                np.stack(
                    [shadow_xz_y_neg * (n - i) / n + corner0 * i / n for i in range(n)],
                    axis=2,
                ),
                np.stack(
                    [shadow_xz_y_neg * (n - i) / n + corner1 * i / n for i in range(n)],
                    axis=2,
                ),
            ],
            axis=2,
        )

        shadow = np.concatenate(
            [
                shadow_xy_z_pos,
                shadow_xy_z_neg,
                shadow_yz_x_pos,
                shadow_yz_x_neg,
                shadow_xz_y_pos,
                shadow_xz_y_neg,
            ],
            axis=2,
        )

    color = np.array(color, dtype=np.float32).reshape(1, 3).repeat(num_boxes * max(1, shadow_density) * 12, axis=0)

    for i in range(num_envs):
        gym.add_lines(viewer, envs[i], num_boxes * 12, skeleton[i], color)
        if shadow_density > 0:
            gym.add_lines(viewer, envs[i], num_boxes * shadow_density * 6 * 2, shadow[i], color)


def print_observation_space(observation_space: Sequence[ObservationSpec]) -> None:
    """Print the observation space to terminal.

    Args:
        observation_space (Sequence[ObservationSpec]): The observation space.
    """
    from itertools import cycle

    from rich.console import Console
    from rich.table import Table

    columns = ["name", "tags", "#dim", "start", "end"]
    console = Console()
    table = Table(*columns, title="Observation Space", show_header=True, header_style="bold magenta", width=120)

    # define different color for different tags
    tags = set([tag for spec in observation_space for tag in spec.tags])
    colors = cycle(["red", "green", "blue", "yellow", "magenta", "cyan"])
    color_map = {tag: next(colors) for tag in tags}

    current = 0
    for spec in observation_space:
        tag = ", ".join([f"[{color_map[tag]}]{tag}[/]" for tag in spec.tags])
        table.add_row(spec.name, tag, str(spec.dim), str(current), str(current + spec.dim))
        current += spec.dim
    console.print(table)


def print_action_space(action_space: Sequence[ActionSpec]) -> None:
    """Print the action space to terminal.

    Args:
        action_space (Sequence[ActionSpec]): The action space.
    """
    from rich.console import Console
    from rich.table import Table

    columns = ["name", "#dim", "start", "end"]
    console = Console()
    table = Table(*columns, title="Action Space", show_header=True, header_style="bold magenta", width=120)

    current = 0
    for spec in action_space:
        table.add_row(spec.name, str(spec.dim), str(current), str(current + spec.dim))
        current += spec.dim
    console.print(table)


def get_action_indices(action_space: Sequence[ActionSpec], device: torch.device = "cpu") -> Tensor:
    """Get action indices.

    Args:
        action_space (Sequence[ActionSpec]): The action space.
    """
    arm_trans_action_dim = 0
    arm_rot_action_dim = 0
    arm_roll_action_dim = 0
    hand_action_dim = 0

    for spec in action_space:
        if "wrist_translation" in spec.name:
            arm_trans_action_dim += spec.dim
        elif "wrist_rotation" in spec.name:
            arm_rot_action_dim += spec.dim
        elif "wrist_3_joint" in spec.name:
            arm_roll_action_dim += spec.dim
        elif "hand" in spec.name:
            hand_action_dim += spec.dim

    arm_trans_action_indices = torch.arange(0, arm_trans_action_dim, device=device)
    arm_rot_action_indices = torch.arange(
        arm_trans_action_dim, arm_trans_action_dim + arm_rot_action_dim, device=device
    )
    arm_roll_action_indices = torch.arange(
        arm_trans_action_dim + arm_rot_action_dim,
        arm_trans_action_dim + arm_rot_action_dim + arm_roll_action_dim,
        device=device,
    )
    hand_action_indices = torch.arange(
        arm_trans_action_dim + arm_rot_action_dim + arm_roll_action_dim,
        arm_trans_action_dim + arm_rot_action_dim + arm_roll_action_dim + hand_action_dim,
        device=device,
    )
    return arm_trans_action_indices, arm_rot_action_indices, arm_roll_action_indices, hand_action_indices


def print_dof_properties(gym, asset, properties: np.ndarray, asset_name: str = ""):
    from rich.console import Console
    from rich.table import Table

    columns = list(properties.dtype.names)
    title = "DOF Properties" + (f" ({asset_name})" if asset_name != "" else "")
    console = Console()
    table = Table("name", *columns, title=title, show_header=True, header_style="bold magenta", width=120)

    for i in range(properties.shape[0]):
        name = gym.get_asset_dof_name(asset, i)
        item = [name] + [str(properties[column][i]) for column in columns]
        table.add_row(*item)

    console.print(table)


def print_links(gym: gymapi.Gym, asset, asset_name: str = ""):
    from rich.console import Console
    from rich.table import Table

    title = "Links" + (f" ({asset_name})" if asset_name != "" else "")
    console = Console()
    table = Table("name", "index", title=title, show_header=True, header_style="bold magenta", width=120)

    for i in range(gym.get_asset_rigid_body_count(asset)):
        name = gym.get_asset_rigid_body_name(asset, i)
        table.add_row(name, str(i))

    console.print(table)


def print_dofs(gym: gymapi.Gym, asset, asset_name: str = ""):
    from rich.console import Console
    from rich.table import Table

    title = "DOFs" + (f" ({asset_name})" if asset_name != "" else "")
    console = Console()
    table = Table("name", "index", title=title, show_header=True, header_style="bold magenta", width=120)

    for i in range(gym.get_asset_dof_count(asset)):
        name = gym.get_asset_dof_name(asset, i)
        table.add_row(name, str(i))

    console.print(table)


def print_links_and_dofs(gym: gymapi.Gym, asset, asset_name: str = ""):
    print_links(gym, asset, asset_name)
    print_dofs(gym, asset, asset_name)


def print_asset_options(asset_options: gymapi.AssetOptions, asset_name: str = ""):
    from rich.console import Console
    from rich.table import Table

    attrs = [
        "angular_damping",
        "armature",
        "collapse_fixed_joints",
        "convex_decomposition_from_submeshes",
        "default_dof_drive_mode",
        "density",
        "disable_gravity",
        "fix_base_link",
        "flip_visual_attachments",
        "linear_damping",
        "max_angular_velocity",
        "max_linear_velocity",
        "mesh_normal_mode",
        "min_particle_mass",
        "override_com",
        "override_inertia",
        "replace_cylinder_with_capsule",
        "tendon_limit_stiffness",
        "thickness",
        "use_mesh_materials",
        "use_physx_armature",
        "vhacd_enabled",
    ]  # vhacd_params
    vhacd_attrs = [
        "alpha",
        "beta",
        "concavity",
        "convex_hull_approximation",
        "convex_hull_downsampling",
        "max_convex_hulls",
        "max_num_vertices_per_ch",
        "min_volume_per_ch",
        "mode",
        "ocl_acceleration",
        "pca",
        "plane_downsampling",
        "project_hull_vertices",
        "resolution",
    ]

    title = "Asset Options" + (f" ({asset_name})" if asset_name != "" else "")
    console = Console()
    table = Table("name", "value", title=title, show_header=True, header_style="bold magenta", width=120)

    for attr in attrs:
        table.add_row(attr, str(getattr(asset_options, attr)) if hasattr(asset_options, attr) else "--")
        if attr == "vhacd_enabled" and hasattr(asset_options, attr) and getattr(asset_options, attr):
            for vhacd_attr in vhacd_attrs:
                table.add_row(
                    f"vhacd_param: {vhacd_attr}",
                    str(getattr(asset_options.vhacd_params, vhacd_attr))
                    if hasattr(asset_options.vhacd_params, vhacd_attr)
                    else "--",
                )

    console.print(table)


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


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def images_to_video(path, images, fps=10, size=(256, 256), suffix="mp4"):
    path = path + f".{suffix}"
    out = cv2.VideoWriter(filename=path, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size, isColor=True)
    for item in images:
        item = cv2.resize(item, size)
        out.write(item.astype(np.uint8))
    out.release()
