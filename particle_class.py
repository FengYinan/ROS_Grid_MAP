# -*- coding: utf-8 -*-

from numpy.random import normal
import numpy as np
from math import sin,cos
from helper_func import *
from resampling import resampling
#from skimage import draw

class particle:

    def __init__(self,num_particles, prior_pos, map_size = 200, random=True):
        self.weight = 1.0/num_particles
        prior = 0.5
        self.gridSize = 0.05
        self.map_size = map_size

        self.pose = prior_pos
        if random:
            self.pose += np.random.normal(0,0.5,3)

        self.grid_map = np.ones((map_size, map_size))
        self.weight = 0.0
        self.probOcc = 0.9
        self.probFree = 0.35
        self.logOddsPrior = prob_to_logodds(prior)

        self.grid_map = self.logOddsPrior * self.grid_map

    def sample_motion_model(self,ut):
        # Inputs : ut = [v w]'
        # x_pre = [x y theta]'
        # v = 1
        # w = 0
        #[v,w] = ut
        # a1 = 0.01
        # a2 = 0.01
        # a3 = 0.01
        # a4 = 0.01
        # a5 = 0.01
        # a6 = 0.01
        # v_cap = v + sample(a1*v**2 + a2*w**2)
        # w_cap = w + sample(a3*v**2 + a4*w**2)
        # gamma_cap = sample(a5*v**2 + a6*w**2)
        # x_prime = x - (v_cap/w_cap)*sin(theta) + v_cap/w_cap*sin(theta + w_cap*del_t)
        # y_prime = y + (v_cap/w_cap)*cos(theta) - v_cap/w_cap*cos(theta + w_cap*del_t)
        # theta_prime = theta + w_cap*del_t + gamma_cap*del_t
        [x, y, theta] = self.pose
        del_t = 0.1
        x_prime = x + cos(theta) * del_t * ut[0] + np.random.normal(0,0.1)
        y_prime = y + sin(theta) * del_t * ut[0] + np.random.normal(0,0.1)
        theta_prime = theta + del_t * ut[1]  + np.random.normal(0,0.3)
        xt = [x_prime, y_prime, theta_prime]
        return xt

    def inv_sensor_model(self,scan):
        mapUpdate = np.zeros(np.shape(self.grid_map))
        robMapPose = pose_world_to_map(self.pose[:2],self.gridSize)
        robTrans = v2t(self.pose)
        laserEndPnts = laser_to_xy(scan,self.gridSize)
        laserEndPnts = np.dot(robTrans,laserEndPnts) # convert from laser – robot coords
        laserEndPntsMapFrame = laser_world_to_map(laserEndPnts[:2],self.gridSize)

        freeCells = []
        for col in range(len(laserEndPntsMapFrame[0])):
            P = np.array(list(bresenham(robMapPose[0], laserEndPntsMapFrame[0,col],robMapPose[1], laserEndPntsMapFrame[1,col])), dtype=np.int).transpose()
            freeCells.append(P)

        for freeCell in freeCells:
            mapUpdate[freeCell[0]][freeCell[1]] = prob_to_logodds(self.probFree)
        for endPnt in laserEndPntsMapFrame.transpose():
            if np.linalg.norm(endPnt) < scan.range_max:
                mapUpdate[endPnt[0]][endPnt[1]] = prob_to_logodds(self.probOcc)
        return mapUpdate, robMapPose, laserEndPntsMapFrame

    def observation_model(self,scan):
        zt = np.array(scan.ranges)
        x, y = pose_world_to_map(self.pose[:2], self.gridSize)
        pose = np.array([x, y, self.pose[2]])
        if x >= self.map_size or y >= self.map_size:
            return 0
        d = self.raytrace(scan,pose)
        weight = 1.0/(np.linalg.norm(zt-d) + 1)
        return weight

    def raytrace(self,scan,pos):
        thresh_occ = 0.75
        x, y, theta = pos[0], pos[1], pos[2]
        ang_min = theta + scan.angle_min
        ang_max = theta + scan.angle_max
        angle_increment = scan.angle_increment

        distances = []

        for i in np.arange(ang_min, ang_max+angle_increment, angle_increment):
            for n in range(1, int(scan.range_max / self.gridSize)):
                x_pos = int(np.round(x + n * self.gridSize * np.cos(i)))
                y_pos = int(np.round(y + n * self.gridSize * np.sin(i)))

                if x_pos >= self.map_size or y_pos >= self.map_size:
                    distances.append(np.infty)
                    break
                if self.grid_map[x_pos][y_pos] > thresh_occ:
                    distances.append(np.linalg.norm([x_pos - x, y_pos - y]))
                    break
                elif n == int(scan.range_max / self.gridSize)-1:
                    distances.append(np.infty)

        return np.array(distances)

    def update(self, ut, scan):
        self.pose = self.sample_motion_model(ut)
        self.weight = self.observation_model(scan)
        [mapUpdate, _, _] = self.inv_sensor_model(scan)
        self.grid_map -= self.logOddsPrior
        self.grid_map += mapUpdate

if __name__ == '__main__':
    num_particles = 30
    # particles = [particle(num_particles) for i in range(num_particles)]
    # while(not ut.end):
    # ut, scan = Subscribe to topics to get twist and laser scan after delta_t
    #     for i in range(num_particles):
    #         particles[i].pose = particles[i].sample_motion_model(ut)
    #         particles[i].weight = particles[i].observation_model(scan)
    #         [mapUpdate, robPoseMap, laserEndPntsMapFrame] = inverse_sensor_model(scan)
    #         particles[i].grid_map -= logOddsPrior
    #         particles[i].grid_map += mapUpdate
    #         plot_map(particles[i].grid_map,robPoseMap, laserEndPntsMapFrame, particles[i].gridSize)
    #     particles = resampling(particles)
