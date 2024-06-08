import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import time
import pybullet_data

direct = p.connect(p.GUI)  #, options="--window_backend=2 --render_device=0")
#egl = p.loadPlugin("eglRendererPlugin")

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF('plane.urdf')
p.loadURDF("r2d2.urdf", [0, 0, 1])
p.loadURDF('cube_small.urdf', basePosition=[0.0, 0.0, 0.025])
id1 = p.addUserDebugParameter('_dis', rangeMin = 0, rangeMax = 10 )
id2 = p.addUserDebugParameter('_yaw', rangeMin = -180, rangeMax = 180 )
id3 = p.addUserDebugParameter('_pit', rangeMin = -180, rangeMax = 180 )
while True:
    p.stepSimulation()
    dis = p.readUserDebugParameter(id1)
    yaw = p.readUserDebugParameter(id2)
    pit = p.readUserDebugParameter(id3)
    p.resetDebugVisualizerCamera(dis,yaw,pit,[0,0,1])
    