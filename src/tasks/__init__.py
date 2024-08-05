import os
import sys
from typing import Any, Dict, Optional

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig, OmegaConf

# from .functional_manipulation import ShadowHandFunctionalManipulation
from .functional_manipulation_underarm import ShadowHandFunctionalManipulationUnderarm
# from .shadow_hand import ShadowHand

# isaacgym_task_map["ShadowHandFunctionalManipulation"] = ShadowHandFunctionalManipulation
isaacgym_task_map["ShadowHandFunctionalManipulationUnderarm"] = ShadowHandFunctionalManipulationUnderarm
# isaacgym_task_map["ShadowHand"] = ShadowHand


def print_cfg(d: Dict, indent: int = 0) -> None:
    """Print the environment configuration.

    Args:
        d (dict): The dictionary to print.
        indent (int, optional): The indentation level. Defaults to 0.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + "  |-- {}: {}".format(key, value))


def parse_hydra_config(
    task_name: str = "",
    isaacgymenvs_path: str = "",
    show_cfg: bool = True,
    args: Optional[Any] = None,
    config_file: str = "config",
) -> DictConfig:
    import isaacgym
    import isaacgymenvs
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
    from hydra.types import RunMode

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("task="):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        if task_name and task_name != arg.split("task=")[1].split(" ")[0]:
            print(
                "[WARNING] Overriding task name ({}) with command line argument ({})".format(
                    task_name, arg.split("task=")[1].split(" ")[0]
                )
            )
    # get task name from function arguments
    else:
        if task_name:
            if args is not None:
                args.overrides.append("task={}".format(task_name))
            else:
                sys.argv.append("task={}".format(task_name))
        else:
            raise ValueError(
                "No task name defined. Set task_name parameter or use task=<task_name> as command line argument"
            )

    # get isaacgymenvs path from isaacgymenvs package metadata
    if isaacgymenvs_path == "":
        if not hasattr(isaacgymenvs, "__path__"):
            raise RuntimeError("isaacgymenvs package is not installed")
        isaacgymenvs_path = list(isaacgymenvs.__path__)[0]

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_dir, "cfg")

    # set omegaconf resolvers
    try:
        OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver("if", lambda condition, a, b: a if condition else b)
    except Exception as e:
        pass
    try:
        OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
    except Exception as e:
        pass

    # get hydra config without use @hydra.main
    args = args if args else get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name="load_isaacgymenv", config_search_path=search_path)
    config = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)

    # print config
    if show_cfg:
        print("\nIsaac Gym environment ({})".format(config.task.name))
        print_cfg(omegaconf_to_dict(config.task))

    return config


def create_env_from_config(config: OmegaConf) -> VecTask:
    cfg = omegaconf_to_dict(config.task)

    try:
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
        )
    except TypeError as e:
        env = isaacgym_task_map[config.task.name](
            cfg=cfg,
            rl_device=config.rl_device,
            sim_device=config.sim_device,
            graphics_device_id=config.graphics_device_id,
            headless=config.headless,
            virtual_screen_capture=config.capture_video,  # TODO: check
            force_render=config.force_render,
        )

    return env


def load_isaacgym_env(
    task_name: str = "",
    isaacgymenvs_path: str = "",
    show_cfg: bool = True,
    args: Optional[Any] = None,
) -> VecTask:
    config = parse_hydra_config(task_name=task_name, isaacgymenvs_path=isaacgymenvs_path, show_cfg=show_cfg, args=args)
    env = create_env_from_config(config)
    return env
