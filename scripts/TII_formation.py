#%%
import sys
import os
sys.path.insert(0, os.getcwd())
os.chdir('../')

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation
import bswarm
import json

class Formation:

    def __init__(self, points, order):
        self.points = points
        self.order = order

def plot_formation(F: Formation, name):
    plt.figure()
    ax = plt.axes(projection='3d')
    points = F.points
    order = F.order
    for i in range(points.shape[1]):
        ax.text3D(points[0, i], points[1, i], points[2, i], str([i]))
        ax.plot3D([points[0, i]], [points[1, i]], [points[2, i]], 'r.')
    plt.title(name)
    plt.show()

def scale_formation(form, scale):
    formNew = np.copy(form)
    for i in range(3):
        formNew[i, :] *= scale[i]
    return formNew

def plot_takeoff(form, name):
    plt.figure()
    ax = plt.axes()
    F=form.points
    ax.plot(F[0, :], F[1, :], 'ro')
    plt.title(name)
    ax.set_xlabel('x---> (m)', fontsize=12,color='blue')
    ax.set_ylabel('y--> (m)', fontsize=12,color='green')
    ax.arrow(0, 0, 0, 0.3,head_width = 0.03,width = 0.008,ec ='green')
    ax.arrow(0, 0, 0.3, 0,head_width = 0.03,width = 0.008,ec ='blue')
    ax.axis('equal')
    for i in range(F.shape[1]):
        ax.text(F[0, i], F[1, i], str(i))
    plt.show()

#%% takeoff
formations = {}
formations['takeoff'] = Formation(
    points = np.array([
        [-0.5,0,0],
        [0,0,0],
        [0.5,0,0],
        [-0.5,1,0],
        [0,1,0],
        [0.5,1,0],
        [-0.5,2,0],
        [0,2,0],
        [0.5,2,0],
        [-1.25,0.75,0],
        [-1.25,1.5,0]
        ]).T,
    order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plot_takeoff(formations['takeoff'], 'takeoff')

#%% TII
letter_scale = np.array([1.5, 1.5, 1.5])

formations['TII'] = Formation(
    points=scale_formation(np.array([
    [-0.5,-0.75,2], [0,0,1.5], [0.5,0,1], 
    [-0.5,2,2], [0,2,1.5], [0.5,2,1], 
    [-0.5,3,2],[0,3,1.5] , [0.5,3,1],[-0.5,0,2],[-0.5,0.75,2]
    ]).T, letter_scale),
    order=[0,9,10,1, 2, 3, 4, 5, 6, 7, 8])

plot_formation(formations['TII'], 'TII')


#%%
class Letters:

    rgb = {
        'black': [0, 0, 0],
        'gold': [255, 100, 15],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'white': [255, 255, 255]
    }

    def __init__(self):
        self.waypoints = [formations['takeoff'].points]
        self.T = []
        self.delays = []
        self.colors = []

    def add(self, formation_name: str, color: str, duration: float, led_delay: float, angle: float=0):
        formation = formations[formation_name]
        assert led_delay*len(formation.order) < duration
        self.T.append(duration)
        self.waypoints.append(bswarm.formation.rotate_points_z(formation.points, angle))
        delay = []
        order = np.array(formation.order)
        delay = np.zeros(len(formation.order))
        for i, drone in enumerate(formation.order):
            delay[drone] = i*led_delay
        self.delays.append(delay.tolist())
        self.colors.append(self.rgb[color])
        
    def plan_trajectory(self, origin):
        trajectories = []
        waypoints = np.array(self.waypoints)
        assert len(waypoints) < 33
        for drone in range(waypoints.shape[2]):
            pos_wp = waypoints[:, :, drone] + origin
            yaw_wp = np.zeros((pos_wp.shape[0], 1))
            traj = tgen.min_deriv_4d(4, 
                np.hstack([pos_wp, yaw_wp]), self.T, stop=False)
            trajectories.append(traj)
        assert len(trajectories) < 32
        return trajectories

def plan_TII():
    letters = Letters()
    letters.add('takeoff', color='blue', duration=2, led_delay=0)
    letters.add('takeoff', color='blue', duration=5, led_delay=0)

    letters.add('TII', color='blue', duration=10, led_delay=0)
    letters.add('TII', color='gold', duration=10, led_delay=0.5)

    letters.add('takeoff', color='blue', duration=5, led_delay=0)
    letters.add('takeoff', color='blue', duration=5, led_delay=0)
    #letters.add('takeoff', color='white', duration=10, led_delay=0.5)
    

    trajectories = letters.plan_trajectory(origin=np.array([0, -2, 1]))
    traj_json = tgen.trajectories_to_json(trajectories)
    data = {}
    for key in traj_json.keys():
        data[key] = {
            'trajectory': traj_json[key],
            'T': letters.T,
            'color': letters.colors,
            'delay': [d[key] for d in letters.delays]
        }
    data['repeat'] = 1
    return trajectories, data

trajectories, data = plan_TII()
print(os.getcwd())
with open('scripts/data/tii_form.json', 'w') as f:
    json.dump(data, f)

print("Hello")
tgen.plot_trajectories(trajectories)
plt.show()

tgen.gazebo_animate_trajectories("gazebo_tii.world",trajectories)


#%%
plt.figure()
tgen.plot_trajectories_time_history(trajectories)
plt.show()

#%%
plt.figure()
tgen.plot_trajectories_magnitudes(trajectories)
plt.show()

#%%
print('number of segments', len(trajectories[0].coef_array()))
#%%
plt.figure()
plt.title('durations')
plt.bar(range(len(data[0]['T'])), data[0]['T'])
plt.show()

#%%

tgen.animate_trajectories('TII_formation.mp4', trajectories, fps=5)


#%%
