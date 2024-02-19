import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    computes the pose matrix (camera matrix) P given 2D and 3D
    points.
    
    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]
    """
    N = X.shape[1]

    Mat = np.zeros((2 * N, 12))

    for i in range(N):
        X_1 = X[0,i]
        Y_1 = X[1,i]
        Z_1 = X[2,i]
        x_2 = x[0,i]
        y_2 = x[1,i]

        Mat[2*i-1,:] = [-X_1,-Y_1, -Z_1, -1, 0,0,0,0, x_2*X_1, x_2*Y_1,x_2*Z_1,x_2]
        Mat[2*i,:] = [0,0,0,0,-X_1,-Y_1,-Z_1, -1, y_2*X_1, y_2*Y_1, y_2*Z_1,y_2]

    U,S,V = np.linalg.svd(Mat)

    P = V[-1,:]
    P = P.reshape((3,4))
    
    
    return P
