import numpy as np
from isaacgym import gymapi
from isaacgymenvs.tasks.base import vec_task
from sim_web_visualizer.isaac_visualizer_client import bind_visualizer_to_gym, create_isaac_visualizer, set_gpu_pipeline


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


# create class for visualizer
class Visualizer:
    def __init__(self, port):
        self.port = port
        # Reload VecTask function to create a hook for sim_web_visualizer
        vec_task.VecTask.create_sim = wrapped_create_sim
        # Create web visualizer
        self.visualizer = create_isaac_visualizer(
            port=port,
            host="localhost",
            keep_default_viewer=False,
            max_env=4,
            use_visual_material=False,
        )

    def render(self, w=256, h=256):
        pil_image = self.visualizer.viz.get_image(w=w, h=h)
        img = np.array(pil_image)[:, :, :3][:, :, ::-1]
        return img

    def set_cam_pose(self, cam_pos, target_pos):
        self.visualizer.viz.set_cam_pos(cam_pos)
        self.visualizer.viz.set_cam_target(target_pos)
