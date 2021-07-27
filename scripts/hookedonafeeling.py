#%%
import sys
import os
sys.path.insert(0, os.getcwd())

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
import bswarm.trajectory as tgen
import bswarm.formation as formation
import bswarm
import json

def plot_formation(F, name):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(F[0, :], F[1, :], F[2, :], 'ro')
    plt.title(name)
    ax.set_xlabel('x---> (m)', fontsize=12)
    ax.xaxis.set_label_coords(1.05, -0.025)
    ax.set_ylabel('y--> (m)', fontsize=12,x=2,y=2)
    ax.set_zlabel('z---> (m)', fontsize=12,x=2,y=2)
    for i in range(F.shape[1]):
        ax.text3D(F[0, i], F[1, i], F[2, i], str(i))
    plt.show()

def scale_formation(form, scale):
    formNew = np.copy(form)
    for i in range(3):
        formNew[i, :] *= scale[i]
    return formNew

def plot_takeoff(F, name):
    plt.figure()
    ax = plt.axes()
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



#%% parameters

# defines how the drones are ordered for circles etc.
rotation_order = [4, 3, 2, 5, 0, 1]

# scaling for formation
form_scale = np.array([1.8, 1.8, 1])

# the takeoff formation
formTakeoff = np.array([
    [-0.5, -1, 0],
    [-0.5, 0, 0],
    [-0.5, 1, 0],
    [0.5, -1, 0],
    [0.5, 0, 0],
    [0.5, 1, 0],
]).T
n_drones = formTakeoff.shape[1]
plot_takeoff(formTakeoff,'takeoff')
#%%

plot_formation(formTakeoff, 'takeoff')
points = []
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    print(theta,i_drone)
    points.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0])
formCircle = scale_formation(np.array(points).T, form_scale)
plot_formation(formCircle, 'circle')

#%%
points = []
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    if i_drone % 2 == 0:
        points.append([0.5*np.cos(theta), 0.5*np.sin(theta), -0.5])
    else:
        points.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0.5])
formTriangle = scale_formation(np.array(points).T, form_scale)
plot_formation(formTriangle, 'triangle')


#%%
points = []
for i_drone in rotation_order:
    theta = i_drone*2*np.pi/n_drones
    if i_drone % 2 == 0:
        points.append([0.5*np.cos(theta), 0.5*np.sin(theta), -0.5])
    else:
        points.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0.5])
formTriangle = scale_formation(np.array(points).T, form_scale)
plot_formation(formTriangle, 'triangle')


#%%

class Geometry:

    rgb = {
        'black': [0, 0, 0],
        'gold': [255, 100, 15],
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'white': [255, 255, 255]
    }

    def __init__(self):
        self.waypoints = [formTakeoff]
        self.T = []
        self.delays = []
        self.colors = []

    @staticmethod
    def sin_z(form, t):
        new_form = np.array(form)
        n_drones = form.shape[1]
        for i in rotation_order:
            new_form[2, i] = 0.5*np.sin(t + i*2*np.pi/n_drones)
        return new_form

    def sin_wave(self, form, n, duration, color):
        for t in np.linspace(0, duration, n):
            formSin = self.sin_z(form, t*np.pi/4)
            self.waypoints.append(formSin)
            self.T.append(duration/n)
            self.colors.append(self.rgb[color])
            n_drones = form.shape[1]
            self.delays.append(np.zeros(n_drones).tolist())

    def spiral(self, form, z, n, duration, color):
        for t in np.linspace(0, 1, n):
            rot_form = formation.rotate_points_z(form, t*2*np.pi)
            shift = np.array([[0, 0, z*np.sin(t)]]).T
            self.waypoints.append(rot_form + shift)
            self.T.append(duration/n)
            self.colors.append(self.rgb[color])
            n_drones = form.shape[1]
            self.delays.append(np.zeros(n_drones).tolist())

    def rotate(self, form, n, duration, color):
        for t in np.linspace(0, 1, n):
            rot_form = formation.rotate_points_z(form, t*2*np.pi)
            self.waypoints.append(rot_form)
            self.T.append(duration/n)
            self.colors.append(self.rgb[color])
            n_drones = form.shape[1]
            self.delays.append(np.zeros(n_drones).tolist())

    def goto(self, form, duration, color):
        self.waypoints.append(form)
        self.T.append(duration)
        self.colors.append(self.rgb[color])
        n_drones = form.shape[1]
        self.delays.append(np.zeros(n_drones).tolist())
    
    def plan_trajectory(self):
        trajectories = []
        origin = np.array([0, 0, 2])
        waypoints = np.array(self.waypoints)
        for drone in range(waypoints.shape[2]):
            pos_wp = waypoints[:, :, drone] + origin
            yaw_wp = np.zeros((pos_wp.shape[0], 1))
            traj = tgen.min_deriv_4d(4, 
                np.hstack([pos_wp, yaw_wp]), self.T, stop=False)
            trajectories.append(traj)

        traj_json = tgen.trajectories_to_json(trajectories)
        data = {}
        for key in traj_json.keys():
            data[key] = {
                'trajectory': traj_json[key],
                'T': self.T,
                'color': g.colors,
                'delay': [d[key] for d in g.delays]
            }
        data['repeat'] = 3
        assert len(trajectories) < 32
        return trajectories, data

# create trajectory waypoints
g = Geometry()
#ooga chaka
g.goto(form=formTakeoff, duration=2, color='blue')
g.sin_wave(form=formTakeoff, n=4, duration=7, color='gold')
#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII gotta feeling
g.goto(form=formCircle, duration=2, color='red')
g.spiral(form=formCircle, z=1, n=6, duration=12, color='red')
#less popular part of chorus
g.goto(form=formCircle, duration=2, color='green')
g.rotate(form=formTriangle, n=6, duration=17, color='white')
#before alcohol
g.rotate(form=formCircle, n=6, duration=12, color='gold')
g.goto(form=formTakeoff, duration=2, color='blue')


#g.rotate(form=formCircle, n=7, duration=20)
#g.rotate(form=formTriangle, n=4, duration=12)
#g.spiral(form=formCircle, z=1, n=4, duration=10)
#g.goto(form=formCircle, duration=2)
#g.spiral(form=formCircle, z=1, n=7, duration=28)
#g.goto(form=formCircle, duration=2)
#g.rotate(form=formTriangle, n=5, duration=18)
g.goto(formTakeoff, duration=2, color='green')

##a
#18 secs for next segment of song
##

#%% plan trajectories
trajectories, data = g.plan_trajectory()

with open('scripts/data/HookedOnAFeeling.json', 'w') as f:
    json.dump(data, f)

tgen.plot_trajectories(trajectories)
tgen.animate_trajectories('hooked.mp4', trajectories, 1)

#%%
plt.figure()
tgen.plot_trajectories_time_history(trajectories)
plt.show()

tgen.gazebo_animate_trajectories("HookedOnAFeeling.world",trajectories)

#%%
plt.figure()
tgen.plot_trajectories_magnitudes(trajectories)
plt.show()

#%%
print('number of segments', len(trajectories[0].coef_array()))
#%%
plt.figure()
plt.title('durations')
plt.bar(range(len(g.T)), g.T)
plt.show()


#%%








