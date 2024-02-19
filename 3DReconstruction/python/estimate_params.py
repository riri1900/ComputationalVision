import numpy as np
from scipy.linalg import svd, qr, rq

def estimate_params(P):
    """
    computes the intrinsic K, rotation R, and translation t from
    given camera matrix P.
    
    Args:
        P: Camera matrix
    """
    U, S, V = svd(P)

    c = V[-1,:3] / V[-1,-1]  

    R, K = qr((np.flipud(P[:,:3]).T))

    K = np.flipud(K.T)
    K = np.fliplr(K)
    R = R.T
    R = np.flipud(R)
    
    for n in range(3):
        if K[n, n] < 0:
            K[:, n] = -K[:, n]
            R[n, :] = -R[n, :]
    t = -np.dot(R,c)
    t = t.reshape(-1,1)
    return K, R, t



