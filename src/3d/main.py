#%%
import pybullet as p
import pybullet_data
import sys
import os
import time
import math
import gym
import pybullet_envs_gymnasium

#%%
print(f"{pybullet_envs.getList()}")

#%%
phys = p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setAdditionalSearchPath("./assets/")
# planeId = p.loadURDF("plane.urdf")
# %%
p.disconnect()
# %%
