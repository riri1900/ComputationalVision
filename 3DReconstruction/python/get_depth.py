import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    depthM = np.zeros_like(dispM, dtype=float)
    #optical centers
    c1 = -np.linalg.inv(np.dot(K1,R1)).dot(np.dot(K1,t1))
    c2 = -np.linalg.inv(np.dot(K2,R2)).dot(np.dot(K2,t2))
    #difference
    diff_cent = c1 - c2

    b = np.linalg.norm(diff_cent)
    f = K1[0,0]

    for x in range(dispM.shape[1]):
        for y in range(dispM.shape[0]):
            if dispM[y,x] ==0:
                depthM[y,x]==0
            else:
                depthM[y,x] = (b * f)/(dispM[y,x])

    #print(depthM)
    return depthM

