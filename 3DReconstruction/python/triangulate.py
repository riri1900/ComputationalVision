import numpy as np

def triangulate(P1, pts1, P2, pts2):
    """
    Estimate the 3D positions of points from 2d correspondence
    Args:
        P1:     projection matrix with shape 3 x 4 for image 1
        pts1:   coordinates of points with shape N x 2 on image 1
        P2:     projection matrix with shape 3 x 4 for image 2
        pts2:   coordinates of points with shape N x 2 on image 2

    Returns:
        Pts3d:  coordinates of 3D points with shape N x 3
    """
    num_points = pts1.shape[0]
    #initialize pts3d
    pts3d = np.zeros((num_points, 3))

    P1_1 = P1[0, :]
    P1_2 = P1[1, :]
    P1_3 = P1[2, :]

    P2_1 = P2[0,:]
    P2_2 = P2[1,:]
    P2_3 = P2[2,:]

    #using the 2D points to create 3D points
    for i in range(num_points):
        x = pts1[i,0]
        y = pts1[i,1]
        x2 = pts2[i,0]
        y2 = pts2[i,1]

        arr = np.array([y * P1_3 - P1_2, P1_1 - x * P1_3, y2 * P2_3 - P2_2, P2_1 - x2 * P2_3])

        U , S , V = np.linalg.svd(arr)
        V = V[-1,:] 
        pts3d[i,:]= V[:3]/V[3] 




    return pts3d
