import numpy as np


A = [2, 3]
B = [1, 5]

def produto_escalar(A, B):
    mag_A = np.sqrt(A[0] **2 + A[1]**2)
    mag_B = np.sqrt(B[0] **2 + B[1]**2)

    prod = mag_A * mag_B * np.cos()