import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Read camera poses from text file.
# The format should be RTAB-Map's "RGBD-SLAM".
def readPosesTXT(fpath):
    
    if os.path.isfile(fpath) == False:
        print("Cannot find poses: '%s'"%fpath)
        sys.exit()

    data = np.loadtxt(fpath, dtype='float_', delimiter=' ')
    
    N = data.shape[0]
    eye = np.eye(4, dtype=np.float32)
    eye = np.expand_dims(eye, axis=0)
    poses = np.repeat(eye, N, axis=0)
    
    if data.shape[1] == 8:
    
        # Assuming RGBD-SLAM format
        timestamps = data[:,0]
        qts = data[:,4:]
        tvecs = data[:,1:4]
        
        for i, (qt, t) in enumerate(zip(qts,tvecs)):
            rot = Rotation.from_quat(qt)
            R = rot.as_matrix()
            poses[i,:3,:3] = R
            poses[i,:3,3] = t
            
        return poses, timestamps

    else:
        print("Check the format of the camera poses: '%s'" %fpath)
        sys.exit()