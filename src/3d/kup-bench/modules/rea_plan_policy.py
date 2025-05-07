from typing import Optional
import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rlbench.demo import Demo
from rlbench.backend.observation import Observation

from tqdm import tqdm as progress
from lib.cam_type import CamType

from modules.demo_obs_dataset import DemoObsDataset

class ReasoningPlanningPolicy(nn.Module):
  def __init__(self, 
    action_shape: int, ## shape of output tensor
    cam_type: CamType
  ):
    super(ReasoningPlanningPolicy, self).__init__()


import numpy as np
import open3d as o3d
import torch
from pyrep import PyRep
from pyrep.objects import Camera, Dummy, Shape, VisionSensor
from pyrep.robots.arms.arm import Arm
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# === Configuration ===
NUM_CANDIDATES = 20
VISIBILITY_THRESHOLD = 0.6
TSDF_VOXEL_SIZE = 0.005
TSDF_SDF_TRUNC = 0.02
TSDF_VOLUME_SIZE = 2.0  # meters

# === Perception Modules ===
class Segmenter:
    def __init__(self, device='cpu'):
      self.model = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
      self.device = device

    def segment(self, image):
        """
        image: HxWx3 RGB np.uint8
        returns: list of masks (boolean arrays) and bounding boxes
        """
        img_t: torch.Tensor = F.to_tensor(image).to(self.device)
        
        outputs = self.model([img_t.double()])[0]
        masks = (outputs['masks'] > 0.5).squeeze(1).cpu().numpy()
        boxes = outputs['boxes'].cpu().numpy()
        return masks, boxes

# === TSDF Fusion ===
class TSDFFusion:
    def __init__(self, device=o3d.core.Device('CPU:0')):
        self.tsdf = o3d.integration.ScalableTSDFVolume(
          voxel_length=TSDF_VOXEL_SIZE,
          sdf_trunc=TSDF_SDF_TRUNC,
          color_type=o3d.integration.TSDFVolumeColorType.RGB8
        )

    def integrate_frame(self, rgb, depth, intrinsic, extrinsic):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth)
        self.tsdf.integrate(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False),
            intrinsic, extrinsic)

    def extract_pointcloud(self):
        return self.tsdf.extract_point_cloud()

# === Visibility Checker ===
class VisibilityChecker:
    def __init__(self, voxel_grid: o3d.geometry.PointCloud, camera: VisionSensor):
        self.grid = voxel_grid
        self.camera = camera

    def compute_visibility(self, camera_pose, target_label=1, rays=200):
        # set camera in scene
        self.camera.setpose
        self.camera.set_pose(camera_pose)
        # sample rays in image plane
        width, height = self.camera.get_resolution()
        xs = np.linspace(0, width-1, int(np.sqrt(rays))).astype(int)
        ys = np.linspace(0, height-1, int(np.sqrt(rays))).astype(int)
        visible = 0
        total = 0
        for x in xs:
            for y in ys:
                origin, direction = self.camera.compute_ray(x, y)
                # shoot ray against voxel grid
                # (simplified: check first intersection point label)
                hit = self._ray_cast(origin, direction)
                if hit == target_label:
                    visible += 1
                total += 1
        return visible / total

    def _ray_cast(self, origin, direction):
        # placeholder: intersect against self.grid; return label
        return np.random.choice([0, 1])
      
    ## Filled in version of the above
    def visibility_score(self, camera_pose, target_label=1, rays=200):
      # set camera pose
      self.camera.set_pose(camera_pose)
      width, height = self.camera.get_resolution()
      # sample pixel grid
      xs = np.linspace(0, width-1, int(np.sqrt(rays))).astype(int)
      ys = np.linspace(0, height-1, int(np.sqrt(rays))).astype(int)
      visible = 0
      total = 0
      # camera extrinsic matrix
      extrinsic = self.camera.get_pose().matrix
      intrinsic = self.camera.get_intrinsics()

      for x in xs:
          for y in ys:
              # compute ray origin and direction in world frame
              origin = extrinsic[:3, 3]
              direction = self._pixel_to_world_ray(x, y, intrinsic, extrinsic)
              # cast ray
              ans = self.raycast_scene.cast_ray(
                  origin=o3d.core.Tensor(origin, dtype=o3d.core.Dtype.Float32),
                  direction=o3d.core.Tensor(direction, dtype=o3d.core.Dtype.Float32))
              # geometry_ids is -1 if no hit
              if ans['geometry_ids'].numpy()[0] == self.mesh_id:
                  visible += 1
              total += 1
      return visible / total if total > 0 else 0.0
# === Pose Candidate Sampler ===
class CameraPoseSampler:
    def __init__(self, arm: Arm, camera_link: Dummy):
        self.arm = arm
        self.camera_link = camera_link

    def sample(self, num, radius=0.3, elevation=(0.2, 0.6)):
        cands = []
        for _ in range(num):
            # spherical coords around target
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(elevation[0], elevation[1])
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi)
            pos = np.array([x, y, z])  # relative offset
            ori = np.eye(3)
            try:
                q = self.arm.solve_ik(position=pos, euler=ori.flatten())
                cands.append(q)
            except Exception:
                continue
        return cands

# === Planning & Execution Loop ===
def main():
    pr = PyRep()
    pr.launch(headless=False)
    env = pr.get_scene()
    arm = UR5()
    cam = VisionSensor("wrist_rgb")
    segmenter = Segmenter(device='cuda')
    tsdf = TSDFFusion()
    sampler = CameraPoseSampler(arm, cam)

    # Main loop (simplified)
    while True:
        rgb = cam.capture_rgb()
        depth = cam.capture_depth()
        intrinsic = cam.get_intrinsic_matrix()
        extrinsic = cam.get_pose()

        # Perception
        masks, boxes = segmenter.segment(rgb)
        tsdf.integrate_frame(rgb, depth, intrinsic, extrinsic)
        pc = tsdf.extract_pointcloud()

        # Visibility check
        vis = VisibilityChecker(pc, cam)
        current_pose = cam.get_pose()
        v = vis.compute_visibility(current_pose)
        if v < VISIBILITY_THRESHOLD:
            candidates = sampler.sample(NUM_CANDIDATES)
            best = max(candidates, key=lambda q: vis.compute_visibility(cam.set_joint_positions(q) or cam.get_pose()))
            # Plan and execute
            planner = arm.get_path_planner(method='rrt')
            traj = planner.plan(best)
            arm.execute_path(traj)
        else:
            # call imitation policy here...
            pass

    pr.shutdown()

if __name__ == '__main__':
    main()
