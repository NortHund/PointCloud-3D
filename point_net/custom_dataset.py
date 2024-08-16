''' Dataset for loading custom point clouds in .ply format
    '''

import os
import sys
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import open3d as o3


class POINTCLOUD(Dataset):
    def __init__(self, path, npoints=4096):
        self.path = path
        self.npoints = npoints     # use  None to sample all the points

    def load_cloud(self):
        # assume npoints is very large if not passed
        if not self.npoints:
            self.npoints = 20000

        #points = np.zeros((1, self.npoints, 3))
        pcd = o3.io.read_point_cloud("../../data/Appartment/appartment_cloud.ply")
        points = np.asarray(pcd.points)

        points = self.downsample(points)
        
        # convert to torch
        points = torch.from_numpy(points).type(torch.float32)
        
        # reshape
        points = points.view(1, self.npoints, 3)

        return points

    def downsample(self, points):
        if len(points) > self.npoints:
            choice = np.random.choice(len(points), self.npoints, replace=False)
        else:
            # case when there are less points than the desired number
            choice = np.random.choice(len(points), self.npoints, replace=True)
        points = points[choice, :] 

        return points

    
    @staticmethod
    def random_rotate(points):
        ''' randomly rotates point cloud about vertical axis.
            Code is commented out to rotate about all axes
            '''
        # construct a randomly parameterized 3x3 rotation matrix
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1,              0,                 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi) ]])

        rot_y = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,                 1,                0],
            [-np.sin(theta), 0, np.cos(theta)]])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi),  0],
            [0,              0,                 1]])

        # rot = np.matmul(rot_x, np.matmul(rot_y, rot_z))
        
        return np.matmul(points, rot_z)


    @staticmethod
    def normalize_points(points):
        ''' Perform min/max normalization on points
            Same as:
            (x - min(x))/(max(x) - min(x))
            '''
        points = points - points.min(axis=0)
        points /= points.max(axis=0)

        return points


    def __len__(self):
        return len(self.data_paths)