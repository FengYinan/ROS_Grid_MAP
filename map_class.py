# -*- coding: utf-8 -*-

from numpy.random import normal
import numpy as np
from math import sin,cos
from helper_func import *
from resampling import resampling
import time
from skimage import draw

class maps:

    def __init__(self, prior_pos, map_size = 1000):

        prior = 0.5
        self.gridSize = 0.05
        self.map_size = map_size

        self.pose = prior_pos

        self.grid_map = np.ones((map_size, map_size))

        self.logprobOcc = 0.9
        self.logprobFree = 0.4
        self.logOddsPrior = prob_to_logodds(prior)

        self.grid_map = self.logOddsPrior * self.grid_map

    def inv_sensor_model(self,scan):


        mapUpdate = np.zeros(np.shape(self.grid_map))

        robTrans = v2t(self.pose)

        robMapPose = pose_world_to_map(self.pose, self.gridSize, self.map_size)

        laserEndPnts, index = laser_to_xy(scan,self.gridSize)
        laserEndPnts = np.dot(robTrans,laserEndPnts) # convert from laser – robot coords
        laserEndPntsMapFrame = laser_world_to_map(laserEndPnts,self.gridSize, self.map_size)

        # free_x = np.array([robMapPose[0]])
        # free_y = np.array([robMapPose[1]])

        for col in range(laserEndPntsMapFrame.shape[1]):
            rr, cc = draw.line(robMapPose[0], robMapPose[1], laserEndPntsMapFrame[0,col], laserEndPntsMapFrame[1,col])
            mapUpdate[rr, cc] = -self.logprobFree

            # free_x = np.concatenate([free_x, rr])
            # free_y = np.concatenate([free_y, cc])

        # mapUpdate[free_x][free_y] = prob_to_logodds(self.probFree)

        laserEndPntsMapFrame = laserEndPntsMapFrame[:, index]
        mapUpdate[laserEndPntsMapFrame[0], laserEndPntsMapFrame[1]] = self.logprobOcc

        return mapUpdate, robMapPose, laserEndPntsMapFrame

    def update(self, pose, scan):
        self.pose = np.array(pose)
        [mapUpdate, _, _] = self.inv_sensor_model(scan)
        self.grid_map += mapUpdate




