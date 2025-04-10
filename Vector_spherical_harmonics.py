import numpy as np
import scipy.special as sp

def VectorBasis(l, m, theta, phi):
    Y = sp.sph_harm(m, l, phi, theta)
    dY_dphi = 1j * m * Y
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    lpmv_ml = sp.lpmv(m, l, cos_theta)
    lpmv_m1l = sp.lpmv(m + 1, l, cos_theta)
    prefactor = np.sqrt(((2 * l + 1) * sp.factorial(l - m)) / (4 * np.pi * sp.factorial(l + m))) * np.exp(1j * m * phi)
    dY_dtheta = prefactor * (lpmv_m1l + m * (cos_theta / sin_theta) * lpmv_ml)
    basis = np.zeros((len(theta), 2), dtype=np.complex128)
    basis[:, 0] = dY_dtheta
    basis[:, 1] = -dY_dphi / sin_theta
    return basis

def convert_to_cartesian(theta, phi, data):
    theta_hat = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]
    phi_hat = [-np.sin(phi), np.cos(phi), 0]
    vector_field = np.zeros((len(theta), 3), dtype=np.complex128)
    for i in range(3):
        vector_field[:,i] = data[0:len(theta)]*phi_hat[i] + data[len(theta):2*len(theta)]*theta_hat[i]
    return vector_field


def VectorSampleData(theta, phi, function_max_degree, true_coefficients):
    """Generate sample data as a linear combination of smake this ore efficient
    pherical harmonics."""
    data = np.zeros((2*len(theta)), dtype=np.complex128)
    idx = 0
    for l in range(0, function_max_degree + 1):
        for m in range(-l, l + 1):
            data[0:len(theta)] += true_coefficients[idx] * VectorBasis(l, m, theta, phi)[:, 0]
            data[len(theta):2*len(theta)] += true_coefficients[idx] * VectorBasis(l, m, theta, phi)[:, 1]
        idx += 1
    return data 

def Tangental(v1, v2):
    if np.abs(np.dot(v1, v2)) <= 1e-6:
        return True
    else:
        return False
    
def Tangent_check(n,x,y,z,A):
    logical = []
    for i in range(n):
            v = [x[i], y[i], z[i]]
            logical.append(Tangental(v, A[i,:]))
    return logical
        

def VectorDesignMatrix(theta, phi, max_degree):
    """Generate the design matrix for the least squares fitting."""
    n = len(theta)
    A = np.zeros((2 * n, (max_degree + 1)**2 - 1), dtype=np.complex128)
    idx = 0
    for l in range(1, max_degree + 1):
        for m in range(-l, l + 1):
            basis = VectorBasis(l, m, theta, phi)
            A[:n, idx] = basis[:, 0]
            A[n:, idx] = basis[:, 1]
            idx += 1
    return A

def construct_L(max_degree, power):
    L = np.zeros(((max_degree + 1)**2 - 1, (max_degree + 1)**2 - 1), dtype=np.complex128)
    idx = 0
    for l in range(1, max_degree + 1):
        for m in range(-l, l + 1):
            L[idx][idx] = (l * (l + 1))**(power+1)
            idx += 1
    return L


def Solve_LSQ(max_degree, data, A, e, grad):
    L = construct_L(max_degree, grad)
    coefficients = np.linalg.inv(A.conj().T @ A + e * L ) @ A.conj().T @ data
    return coefficients                

    
    

