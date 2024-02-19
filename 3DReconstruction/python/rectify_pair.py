import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    takes left and right camera paramters (K, R, T) and returns left
    and right rectification matrices (M1, M2) and updated camera parameters. You
    can test your function using the provided script testRectify.py
    """
    # YOUR CODE HERE
    #computing the optical center
    c1 = -np.linalg.inv(np.dot(K1,R1)).dot(np.dot(K1,t1))
    c2 = -np.linalg.inv(np.dot(K2,R2)).dot(np.dot(K2,t2))

    #computing Rotation matrices
    #(a)
    r1 =np.abs(c1 - c2)/np.linalg.norm(c1-c2) 

    #(b)
    r2 = np.cross(R1[2,:], r1)
    r2 = r2 / np.linalg.norm(r2)
    r2_2 = np.cross(R2[2,:],r1)
    r2_2 = r2_2/np.linalg.norm(r2_2)

    #(c)
    r3 = np.cross(r1,r2)
    r3 = r3/ np.linalg.norm(r3) 
    r3_2 = np.cross(r1,r2_2)
    r3_2 = r3_2/np.linalg.norm(r3_2)

    #constructing the rotation matrices
    R1n = np.vstack([r1,r2,r3]).T
    R2n = np.vstack([r1,r2_2,r3_2]).T

    #computing intrinsic parameter
    K1n = K2
    K2n =K2

    #translation mat 
    t1n = -np.dot(R1n,c1)
    t2n = -np.dot(R2n,c2)

    #Rectification matrix
    M1 = np.dot(np.dot(K1n,R1n),np.linalg.inv(np.dot(K1,R1)))
    M2 = np.dot(np.dot(K2n,R2n), np.linalg.inv(np.dot(K2,R2)))

    #M1, M2, K1n, K2n, R1n, R2n, t1n, t2n = None, None, None, None, None, None, None, None

    return M1, M2, K1n, K2n, R1n, R2n, t1n, t2n

