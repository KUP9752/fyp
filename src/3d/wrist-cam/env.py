#%%
import pybullet as p
import time
import pybullet_data
#%%
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.8)

planeId = p.loadURDF("plane.urdf") 
robotStartPos = [0,0,0]

robotId = p.loadURDF("franka_panda/panda.urdf", robotStartPos, useFixedBase = True)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
robotStartPos

# %%
p.disconnect()
# %%
