import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """

    N1 = pts1.shape[0]
    N2 = pts2.shape[0]
    pts1 = np.hstack((pts1, np.ones((N1, 1))))
    pts2 = np.hstack((pts2, np.ones((N2, 1))))

    #Normalization
    pts1_norm = pts1 / M 
    pts2_norm =  pts2 / M 


    x1 = pts1_norm[:,0]
    y1 = pts1_norm[:,1]
    z1 = pts1_norm[:,2]
    x2 = pts2_norm[:,0]
    y2 = pts2_norm[:,1]
    z2 = pts2_norm[:,2]
    
    #Constructing the matrix 

    Mat = np.column_stack((x1 * x2, x1 * y2  , x1*z2, y1 * x2, y1 * y2, y1*z2, z1*x2, z1*y2,z1*z2 ))
    
    #SVD  

    U, S, V = svd(Mat)

    F = V[-1]
    F =  F.reshape(3,3)

    #rank 2 constraint
    U, S, V = svd(F)
    S[2] = 0 
    F = U @ np.diag(S) @ V.T
   
    #refine
    F_ref = refineF(F,pts1_norm,pts2_norm)

    #un-normalize F

    un_norm = np.array([(1/M, 0, 0), (0, 1/M, 0), (0, 0, 1)])
  

    F = un_norm.T @ F_ref @ un_norm    
    return F
