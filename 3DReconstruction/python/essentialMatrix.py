import numpy as np

def essentialMatrix(F, K1, K2):
    """
    Args:
        F:  Fundamental Matrix
        K1: Camera Matrix 1
        K2: Camera Matrix 2   
    Returns:
        E:  Essential Matrix  
    """
    E = np.dot(K2.T,np.dot(F,K1))
    return E