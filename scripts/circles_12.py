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
import bswarm.formation as form
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
    ax.set_xlabel('x---> (m)', fontsize=12)
    ax.xaxis.set_label_coords(1.05, -0.025)
    ax.set_ylabel('y--> (m)', fontsize=12,x=2,y=2)
    ax.set_zlabel('z---> (m)', fontsize=12,x=2,y=2)
    for i in range(points.shape[1]):
        ax.text3D(points[0, i], points[1, i], points[2, i], str(i))
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
        [-0.5, -1, 0],
        [-0.5, 0, 0],
        [-0.5, 1, 0],
        [0.5, -1, 0],
        [0.5, 0, 0],
        [0.5, 1, 0],
         [-1.5, -1, 0],
        [-1.5, 0, 0],
        [-1.5, 1, 0],
        [1.5, -1, 0],
        [1.5, 0, 0],
        [1.5, 1, 0],
        ]).T,
    order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
plot_takeoff(formations['takeoff'], 'takeoff')

plot_formation(formations['takeoff'], 'takeoff')

#%% Circles
rotation_order = [4, 3, 2, 5, 0, 1]

letter_scale = np.array([1.5, 1.5, 1.5])

n_drones = 6
points = []
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    points.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0])
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    points.append([1*np.cos(theta), 1.5*np.sin(theta), 0.5])


points= np.array(points)
print(points[:13, :].shape[1])
formations['Circle'] = Formation(
    points=scale_formation(np.array(points).T, letter_scale),
    order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

plot_formation(formations['Circle'], 'Circle')

#%%

points = []
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    points.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0.5])
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    points.append([1*np.cos(theta), 1.5*np.sin(theta), 0])

print(points[1])
formations['Circle_1'] = Formation(
    points=scale_formation(np.array(points).T, letter_scale),
    order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

plot_formation(formations['Circle_1'], 'Circle_1')
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

def plan_letters(letter_string: str):
    letters = Letters()
    letters.add('takeoff', color='blue', duration=2, led_delay=0)
    letters.add('takeoff', color='blue', duration=5, led_delay=0)
    for i, letter in enumerate(letter_string.split(' ')):
        letters.add(letter, color='blue', duration=5, led_delay=0)
        if i == 0:
            for theta in np.linspace(0, 2*np.pi, 5)[1:]:
                print(theta)
                letters.add(letter, color='gold', duration=4, led_delay=0, angle=theta)
        formation = formations[letter]
        plot_formation(formations[letter], letter)
        a,b= np.hsplit(formation.points,2)
        for j in range(6):
            a = form.rotate_points_z(a,np.pi/3)
            points = np.concatenate((a, b), axis=1)
            formations[str(j)] = Formation(points,order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            plot_formation(formations[str(j)], str(j))
            letters.add(str(j), color='gold', duration=2, led_delay=0)
        for j in range(6):
            b = form.rotate_points_z(b,-np.pi/3)
            points = np.concatenate((a, b), axis=1)
            formations[str(j)] = Formation(points,order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            plot_formation(formations[str(j)], str(j))
            letters.add(str(j), color='gold', duration=2.5, led_delay=0)
        for j in range(6):
            a = form.rotate_points_z(a,np.pi/3)
            b = form.rotate_points_z(b,-np.pi/3)
            points = np.concatenate((a, b), axis=1)
            formations[str(j)] = Formation(points,order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            plot_formation(formations[str(j)], str(j))
            letters.add(str(j), color='gold', duration=2.5, led_delay=0)
    letters.add('takeoff', color='blue', duration=8, led_delay=0)
    

    trajectories = letters.plan_trajectory(origin=np.array([0, 0, 2]))
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

trajectories, data = plan_letters('Circle_1')

with open('scripts/data/circles_12.json', 'w') as f:
      json.dump(data, f)

tgen.plot_trajectories(trajectories)
plt.show()

tgen.gazebo_animate_trajectories("circle_12.world",trajectories)


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

tgen.animate_trajectories('circles_12.mp4', trajectories, fps=5)


#%%
