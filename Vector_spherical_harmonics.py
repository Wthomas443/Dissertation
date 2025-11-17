import numpy as np
import scipy.special as sp
from numba import jit

@jit(nopython=True)
def precompute_trigonometric(theta, phi):
    """Precompute trigonometric functions efficiently"""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    return cos_theta, sin_theta


def VectorBasis(l, m, theta, phi, cos_theta, sin_theta):
    
    Y = sp.sph_harm(m, l, phi, theta)
    dY_dphi = 1j * m * Y
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

def VectorDesignMatrix(theta, phi, max_degree, batch_size=None):
    """Generate the design matrix for the least squares fitting."""
    n = len(theta)
    L = (max_degree + 1)**2 - 1

    cos_theta, sin_theta = precompute_trigonometric(theta, phi)
    
    if batch_size is None:
        A = np.zeros((2 * n, L), dtype=np.complex128)
        idx = 0
        for l in range(1, max_degree + 1):
            for m in range(-l, l + 1):
                basis = VectorBasis(l, m, theta, phi, cos_theta, sin_theta)
                A[:n, idx] = basis[:, 0]
                A[n:, idx] = basis[:, 1]
                idx += 1
        return A
    else:
        return VectorDesignMatrix_Batched(theta, phi, max_degree, batch_size, cos_theta, sin_theta)
    
def VectorDesignMatrix_Batched(theta, phi, max_degree, batch_size, cos_theta, sin_theta):
    """Process design matrix in batches to reduce memory usage"""
    n = len(theta)
    total_coeffs = (max_degree + 1)**2 - 1
    A = np.zeros((2 * n, total_coeffs), dtype=np.complex128)
    
    # Process degrees in batches
    degrees = list(range(1, max_degree + 1))
    num_batches = (len(degrees) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(degrees))
        
        batch_degrees = degrees[start_idx:end_idx]
        
        for l in batch_degrees:
            for m in range(-l, l + 1):
                basis_idx = l**2 + m - 1  # Calculate the correct index
                basis = VectorBasis(l, m, theta, phi, cos_theta, sin_theta)
                A[:n, basis_idx] = basis[:, 0]
                A[n:, basis_idx] = basis[:, 1]
    
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
