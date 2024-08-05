import json
import os
import pickle
import random
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from pytorch3d.ops import sample_farthest_points


class FunctionalGraspingDataset(object):
    def __init__(self):
        pass

    def parse_pose(self, pose: Dict[str, Any]) -> torch.Tensor:
        position = torch.tensor([pose["position"][key] for key in ["x", "y", "z"]]).float()
        quaternion = torch.tensor([pose["quaternion"][key] for key in ["x", "y", "z", "w"]]).float()
        return torch.cat([position, quaternion])


class OakInkDataset(FunctionalGraspingDataset):
    dataset_name: str = "oakink"
    data: Dict[str, Any]
    # fmt: off
    dof_names: List[str] = [
        "rh_WRJ2", "rh_WRJ1",
        'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
        'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
        'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
        'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
        'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',
    ]
    # fmt: on

    def __init__(
        self,
        dataset_dir: str,
        base_prim: str = "palm",
        device: Optional[Union[str, torch.device]] = None,
        pcl_num: int = 1024,
        num_object: Optional[int] = -1,
        num_object_per_category: Optional[int] = -1,
        queries: Optional[Dict[str, Any]] = None,
        *,
        original_categories_statistics_path: str = "data/ori.json",
        metainfo_path: str = "data/oakink_metainfo.csv",
        skipcode_path: str = "data/oakink_skipcode.csv",
        pretrained_embedding: bool = False,
        precomputed_sdf: bool = False,
        pose_level_sampling: bool = False,
    ):
        super().__init__()

        self.data = {}
        self.label_paths = []
        self.device = device
        self.pcl_num = pcl_num
        self.object_num = num_object
        self.max_per_cat = num_object_per_category
        self.object_geo_level = None  # TODO: remove those two lines or set from the queries
        self.object_scale = None
        self.pose_level_sampling = pose_level_sampling
        self.original_category_statistics = json.load(open(original_categories_statistics_path, "r"))

        category_dict = np.load("./data/category.npy", allow_pickle=True).tolist()
        num_full_categories = len(set(category_dict.values()))
        category_matrix = torch.eye(num_full_categories)

        metainfo: pd.DataFrame = pd.read_csv(metainfo_path)

        # filter invalid codes
        if os.path.exists(skipcode_path):
            skipcode: pd.DataFrame = pd.read_csv(skipcode_path)
            invalid_codes = skipcode.query("ret_code != 0")["code"].tolist()
            metainfo = metainfo[~metainfo["code"].isin(invalid_codes)]
        else:
            warnings.warn("Skipcode file not found, skipping invalid code filtering")

        # filter grasping poses
        for key, candidates in queries.items():
            if isinstance(candidates, list) or isinstance(candidates, ListConfig):
                metainfo = metainfo[metainfo[key].isin(candidates)]
            elif candidates is not None:
                metainfo = metainfo[metainfo[key] == candidates]

        if num_object_per_category != -1:
            instances = metainfo[["category", "code"]].drop_duplicates()
            instances = instances.groupby("category")["code"].head(num_object_per_category)
            metainfo = metainfo[metainfo["code"].isin(instances.values)]

        if num_object != -1:
            instances = metainfo["code"].drop_duplicates()
            instances = instances.sample(min(num_object, instances.shape[0]), random_state=42)
            metainfo = metainfo[metainfo["code"].isin(instances.values)]

        if "category" in queries:
            if isinstance(queries["category"], list) or isinstance(queries["category"], ListConfig):
                self.object_cat = "__".join(queries["category"])
            else:
                self.object_cat = queries["category"]
        else:
            self.object_cat = "all"

        for index, row in metainfo.iterrows():
            code, filepath = row["code"], row["filepath"]

            with open(filepath, "r") as f:
                data = json.load(f)

            if code not in self.data:
                self.data[code] = []
            self.data[code].append((data, row))
            self.label_paths.append(filepath)

        self.num_samples = sum([len(self.data[code]) for code in self.data])
        self.num_objects = len(self.data)
        self.categories = []
        self.object_codes = []
        self.grasp_names = []
        self.code_names = []
        self.clutser_ids = []
        self.object_categories = []
        self.category_object_codes = {}
        self.indices = torch.zeros(self.num_objects + 1, dtype=torch.long, device=self.device)
        self.inverse_indices = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        self._joints = torch.zeros(self.num_samples, len(self.dof_names), device=self.device)
        self._object_poses = torch.zeros(self.num_samples, 7, device=self.device)
        self._pointclouds = torch.zeros(self.num_objects, 4096, 3, device=self.device)
        self._category_matrix = torch.zeros(self.num_objects, num_full_categories, device=self.device)
        self._raw_samples = []

        if precomputed_sdf:
            self._sdf_fields = torch.zeros(self.num_objects, 200, 200, 200, device="cpu")

        print(f"Total number of samples: {self.num_samples}")
        print(f"Total number of objects: {self.num_objects}")

        index = 0
        for cur, code in enumerate(self.data):
            self.indices[cur] = index
            for i, (sample, meta) in enumerate(self.data[code]):
                self._raw_samples.append(sample)
                self.code_names.append(meta["code"])
                self.grasp_names.append(meta["pose"])
                self.clutser_ids.append(int(meta["cluster"]))
                self._joints[index] = torch.tensor(
                    [sample["joints"][name] for name in self.dof_names], device=self.device
                )
                self._object_poses[index] = self.parse_pose(sample["object_pose_wrt_palm"])
                self.data[code][i] = index

                if meta["category"] not in self.categories:
                    self.categories.append(meta["category"])
                    self.category_object_codes[meta["category"]] = []
                if meta["code"] not in self.object_codes:
                    self.object_codes.append(meta["code"])
                    self.object_categories.append(meta["category"])
                    self.category_object_codes[meta["category"]].append(meta["code"])
                self.inverse_indices[index] = cur
                index += 1
        self.indices[-1] = index
        self.categories = list(self.categories)
        self.object_codes = list(self.object_codes)
        self.object_categories = list(self.object_categories)
        self.grasp_names = np.array(self.grasp_names)
        self.code_names = np.array(self.code_names)
        self.clutser_ids = np.array(self.clutser_ids)

        obj_pcl_buf_all_path = "data/pcl_buffer_4096_all.pkl"
        with open(obj_pcl_buf_all_path, "rb") as f:
            self.obj_pcl_buf_all = pickle.load(f)

        for i, code in enumerate(self.object_codes):
            self._pointclouds[i] = torch.from_numpy(self.obj_pcl_buf_all[code]).float().to(self.device)
            self._category_matrix[i] = category_matrix[category_dict[code]]
            if precomputed_sdf:
                self._sdf_fields[i] = torch.from_numpy(np.load("data/precomputed_sdf/" + code + ".npy")).float()

        if pretrained_embedding:
            self.embeddings = pd.read_csv("pointnet_pretrain_embeddings.csv")
            self.embeddings.set_index("code", inplace=True)
        else:
            self.embeddings = None

        # support pose-level sampling
        if self.pose_level_sampling:
            self.manipulated_codes = []
            for code in self.object_codes:
                self.manipulated_codes.extend([code] * len(self.data[code]))
            random.seed(42)
            random.shuffle(self.manipulated_codes)
            self._env_counter = {}
        else:
            self.manipulated_codes = deepcopy(self.object_codes)

        print(">>> Oakink Dataset Initialized")

    def resample(self, num_samples: int) -> List[str]:
        """Resample the current dataset to match the original distribution.

        Args:
            num_samples (int): Number of objects to sample

        Returns:
            List[str]: List of object codes
        """
        print("Resampling dataset...")
        print("Categories: ", self.categories)

        names = self.categories.copy()
        counts = [self.original_category_statistics[name] for name in names]
        probs = np.array(counts) / np.sum(counts)

        categories = np.random.choice(names, size=num_samples, p=probs)

        codes = []
        for category in categories:
            codes.append(random.choice(self.category_object_codes[category]))

        return codes

    def get_boundingbox(self, pointclouds: torch.Tensor) -> torch.Tensor:
        """Computes the bounding box of a point cloud.

        Args:
            pointclouds (torch.Tensor): Point cloud tensor of shape (..., N, 3)

        Returns:
            torch.Tensor: Bounding box tensor of shape (..., 6)
        """
        corner_max = torch.max(pointclouds, dim=-2)[0]
        corner_min = torch.min(pointclouds, dim=-2)[0]
        return torch.cat((corner_max, corner_min), dim=-1).to(self.device)

    def sample(self, object_indices: torch.LongTensor) -> Dict[str, Any]:
        if self.pose_level_sampling:
            indices = object_indices
            object_indices = self.inverse_indices[indices]

        else:
            assert object_indices.dim() == 1, "Object indices must be a 1D tensor"
            assert object_indices.dtype == torch.long, "Object indices must be a 1D tensor of longs"
            assert object_indices.max() < self.num_objects and object_indices.min() >= 0, "Object indices out of range"

            lower = self.indices[object_indices].float()
            upper = self.indices[object_indices + 1].float()
            indices = lower + (upper - lower) * torch.rand_like(object_indices, dtype=torch.float, device=self.device)
            indices = torch.floor(indices).long()
            indices = torch.min(indices, self.indices[-1] - 1)

        pointclouds = self._pointclouds[object_indices]
        pointclouds = sample_farthest_points(pointclouds, K=self.pcl_num)[0]
        boundingbox = self.get_boundingbox(pointclouds)
        category_onehot = self._category_matrix[object_indices]
        grasp = self.grasp_names[indices.detach().cpu().numpy()]
        code = self.code_names[indices.detach().cpu().numpy()]
        clutser_ids = self.clutser_ids[indices.detach().cpu().numpy()]

        return {
            "joints": self._joints[indices],
            "pose": self._object_poses[indices],
            "pointcloud": pointclouds,
            "index": indices,
            "object_index": object_indices,
            "bbox": boundingbox,
            "category_onehot": category_onehot,
            "grasp": grasp,
            "code": code,
            "cluster": clutser_ids,
        }

    def get_object_index(self, object_code: str) -> int:
        if self.pose_level_sampling:
            curr = self._env_counter.get(object_code, 0)
            self._env_counter[object_code] = curr + 1
            index = self.object_codes.index(object_code)
            count = self.indices[index + 1] - self.indices[index]
            return self.indices[index] + (curr % count)
        else:
            return self.object_codes.index(object_code)


def compute_implicit_sdf(
    vertices: torch.Tensor, faces: torch.Tensor, grid_size: int = 200, space: float = 0.5
) -> torch.Tensor:
    """Computes the Signed Distance Field (SDF) for a given 3D mesh represented by vertices and faces.

    The default grid size and space size are chosen to match this paper:
        https://arxiv.org/abs/2211.10957

    Args:
        vertices (torch.Tensor): Tensor of shape (N, 3) representing the 3D coordinates of the mesh vertices.
        faces (torch.Tensor): Tensor of shape (M, 3) representing the indices of the vertices that form each face of the mesh.
        grid_size (int, optional): The number of grid points along each axis. Defaults to 200.
        space (float, optional): The size of the space in which the mesh is defined. Defaults to 0.5.

    Returns:
        sdf_field (torch.Tensor): Tensor of shape (grid_size, grid_size, grid_size) representing the SDF field.
            - Positive values indicate that the point is outside the mesh.
            - Negative values indicate that the point is inside the mesh.
    """
    import kaolin as kal

    assert vertices.dim() == 2 and vertices.shape[1] == 3, "vertices must be of shape (N, 3)"
    assert faces.dim() == 2 and faces.shape[1] == 3, "faces must be of shape (M, 3)"
    assert vertices.device == faces.device, "vertices and faces must be on the same device"
    device = vertices.device

    unit = 2 * space / grid_size
    axis = torch.linspace(-space + unit / 2, space - unit / 2, grid_size, device=device)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(device)

    grid_points = grid_points.unsqueeze(0)
    vertices = vertices.unsqueeze(0)

    face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)

    squared_distance, _, _ = kal.metrics.trianglemesh.point_to_mesh_distance(grid_points, face_vertices)
    distance = torch.sqrt(squared_distance)

    sign = kal.ops.mesh.check_sign(vertices, faces, grid_points)

    sdf = distance * torch.where(sign, -1.0, 1.0)
    sdf_field = sdf.reshape(grid_size, grid_size, grid_size)
    return sdf_field


def point_to_mesh_distance(
    points: torch.Tensor, sdf: torch.Tensor, indices: torch.LongTensor, space: float = 0.5
) -> torch.Tensor:
    """Calculates the distance from each point to the mesh represented by the signed distance field (SDF).

    Args:
        points (torch.Tensor): Tensor of shape (B, N, 3) representing the coordinates of N points.
        sdf (torch.Tensor): Tensor of shape (M, D, D, D) representing the signed distance field.
        indices (torch.LongTensor): Tensor of shape (B,) representing the indices of the mesh elements corresponding to each point.
        space (float, optional): The spacing between grid cells in the SDF. Defaults to 0.5.

    Returns:
        torch.Tensor: Tensor of shape (B, N) representing the distance from each point to the mesh.
    """
    assert points.shape[-1] == 3
    assert points.dim() == 3 or points.dim() == 2, "points must be of shape (B, N, 3) or (N, 3)"
    assert sdf.dim() == 4 and sdf.shape[1] == sdf.shape[2] == sdf.shape[3], "sdf must be of shape (M, D, D, D)"
    assert indices.dim() == 1 and indices.dtype == torch.long, "indices must be a 1D tensor of longs"
    assert indices.shape[0] == points.shape[0], "indices must have the same batch size as points"
    assert indices.min() >= 0 and indices.max() < sdf.shape[0], "indices out of range"

    device, sdf_device = points.device, sdf.device
    points, indices = points.to(sdf_device), indices.to(sdf_device)

    ndim = points.dim()
    batch_size = points.shape[0]
    grid_size = sdf.shape[-1]

    points = points.view(batch_size, -1, 3)
    num_points = points.shape[1]

    coords = (points + space) / (2 * space) * grid_size
    coords = coords.floor().clamp(0, grid_size - 1).long()

    # Get the SDF values at the corresponding coordinates
    indices = indices.unsqueeze(1).expand(-1, num_points)
    coords_x, coords_y, coords_z = coords.unbind(dim=-1)
    sdf_values = sdf[indices, coords_x, coords_y, coords_z]

    return (sdf_values if ndim == 3 else sdf_values.squeeze(1)).to(device)
